# SPDX-FileCopyrightText: Copyright (c) 2022 Phil Underwood for Underwood Underground
#
# SPDX-License-Identifier: MIT
"""
Calibration class: main class to use
"""
from collections import namedtuple

from . import nm
from .rbf import RBF
from .sensor import Sensor, SensorError
from .utils import normalise, solve_least_squares, cross

try:
    from typing import Tuple, Dict, Optional
except ImportError:
    # ignore if running in CircuitPython/MicroPython
    pass

try:
    # work with normal numpy
    import numpy as np

    arccos = np.arccos
except ImportError:
    # or work with ulab if we are in CircuitPython
    from ulab import numpy as np

    arccos = np.acos

__version__ = "0.0.0+auto.0"
__repo__ = "https://github.com/furbrain/CircuitPython_mag_cal.git"


def _vector_from_matrices(matrix: np.ndarray, i: int, j: int):
    return np.array([x[i, j] for x in matrix])


class CalibrationError(Exception):
    """
    Some sort of error from the Calibration Module
    """


class DipAnomalyError(CalibrationError):
    """
    Dip is not what we would expect it to be
    """


class MagneticAnomalyError(CalibrationError):
    """
    Magnetic field strength is not what we would expect it to be
    """


class GravityAnomalyError(CalibrationError):
    """
    Gravity field strength is not what we would expect it to be - device moving during shot?
    """


Strictness = namedtuple("Strictness", ("mag", "grav", "dip"))


class Calibration:
    """
    Object representing a magnetometer and accelerometer calibration
    """

    MAGNETOMETER = 1
    ACCELEROMETER = 2
    BOTH = MAGNETOMETER | ACCELEROMETER

    OFF = 0
    SOFT = 1
    HARD = 2

    ELLIPSOID = 0
    """Fit to Ellipsoid"""
    AXIS_CORRECTION = 1
    """Fit to ellipsoid then correct any axis misalignment"""
    NON_LINEAR = 2
    """as per `AXIS_CORRECTION` and then do correction for non-linear effects"""
    FAST_NON_LINEAR = 3
    """as per `AXIS_CORRECTION` and then do quick non-linear correction"""

    _DEFAULT_STRICTNESS = Strictness(grav=2.0, mag=2.0, dip=3.0)

    def __init__(self, mag_axes: str = "+X+Y+Z", grav_axes: str = None):
        """
        Create an object representing the calibration coefficients for a combined
        magnetometer and accelerometer

        :param str mag_axes: A string representing how your magnetometer is mounted with respect to
          your device. For each axis XYZ of your device (see above for axis descriptions), state the
          corresponding axis of your sensor. Add a ``+`` or ``-`` to let us know if it is inverted.
          So for a sensor that is correctly mounted it will be ``"+X+Y+Z"``. If the sensors Y axis
          points to the left of the device, the X is forwards and Z is down, specify this as
          ``"-Y+X-Z"``
        :param str grav_axes: Same format as ``mag_axes`` but for the accelerometer. Default is copy
          of mag_axes.
        """
        if grav_axes is None:
            grav_axes = mag_axes
        self.mag = Sensor(axes=mag_axes)
        self.grav = Sensor(axes=grav_axes)
        self.dip_avg: Optional[float] = None

    def calibrate(
        self,
        mag_data: np.ndarray,
        grav_data: np.ndarray,
        routine: int = FAST_NON_LINEAR,
    ):
        """
        Perform a full calibration, with an algorithm depending on the value of ``routine``. If
        you select a routine other than `ELLIPSOID` you must provide at least one run
        of at least four shots in the same direction with varying amonts of roll. Ideally two sets
        of eight readings, but this is not vital.

        :param np.ndarray mag_data: Numpy array of magnetic readings of shape (N,3)
        :param np.ndarray grav_data: Numpy array of gravity readings of shape (M,3)
        :param routine: what level of calibration to perform:

          * `ELLIPSOID`: Simplest form of calibration, very fast, does not require any sets of
            readings to be aligned. Does not correct for misalignment between pointer and sensors.
          * `AXIS_CORRECTION`: This routine performs the `ELLIPSOID` and then applies a
            rotation to offset any misalignment between the pointer and sensors (and also
            misalignment between accelerometer and magnetometer if relevant). This process will
            automatically identify which shots have been taken in the same direction
          * `NON_LINEAR`: Performs calibration as per `AXIS_CORRECTION`, then uses an
            optimisation process to account for non-linear sensor response. See `fit_non_linear`
            for details.
          * `FAST_NON_LINEAR`: Performs calibration as per `AXIS_CORRECTION`, then uses a
            least-squares process to account for non-linear sensor response. A lot faster than
            `NON_LINEAR`, but slightly less accurate. See `fit_non_linear_quick` for details

        :return: Measure of error: percentage error of fit for `ELLIPSOID`,
          standard deviation of error in degrees for other methods. Normally <1 degree is
          acceptable, <0.5 degrees is good.
        """
        self.fit_ellipsoid(mag_data, grav_data)
        if routine >= self.AXIS_CORRECTION:
            runs = self.find_similar_shots(mag_data, grav_data)
            if len(runs) == 0:
                raise ValueError("No runs of shots all in the same direction found")
            paired_data = [(mag_data[a:b], grav_data[a:b]) for a, b in runs]
            self.fit_to_axis(paired_data)
            if routine == self.NON_LINEAR:
                self.fit_non_linear(paired_data)
            elif routine == self.FAST_NON_LINEAR:
                self.fit_non_linear_quick(paired_data)
            self.set_field_characteristics(mag_data, grav_data)
            return self.accuracy(paired_data)
        # just ellipsod fit done, so use uniformity measure
        self.set_field_characteristics(mag_data, grav_data)
        return np.mean(self.uniformity(mag_data, grav_data))

    def get_angles(self, mag, grav) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get device azimuth(bearing), inclination, and roll, given the magnetic and gravity readings

        :param np.ndarray mag: Magnetic readings, either as numpy array or sequence of 3 floats
        :param np.ndarray grav: Gravity readings, either as numpy array or sequence of 3 floats
        :return: (azimuth, inclination, roll) in degrees
        """

        matrix = self.get_orientation_matrix(mag, grav)
        azimuth, inclination, roll = self.matrix_to_angles(matrix)
        return azimuth, inclination, roll

    def as_dict(self) -> Dict:
        """
        Convert the current calibration to a dictionary, suitable for serialising via json.

        :return: The calibration as a dictionary
        :rtype: dict
        """
        dct = {
            "mag": self.mag.as_dict(),
            "grav": self.grav.as_dict(),
            "dip_avg": self.dip_avg,
        }
        return dct

    @classmethod
    def from_dict(cls, dct: Dict) -> "Calibration":
        """
        Create a Calibration object based on a dict previously produced by `as_dict`

        :param dict dct: dict to instantiate
        :return: Calibration object
        """
        instance = cls()
        instance.mag = Sensor.from_dict(dct["mag"])
        instance.grav = Sensor.from_dict(dct["grav"])
        instance.dip_avg = dct["dip_avg"]
        return instance

    def fit_ellipsoid(
        self, mag_data: np.ndarray, grav_data: np.ndarray
    ) -> Tuple[float, float]:
        """
        Take multiple sets of readings in various directions. You can then use this function
        to determine an ideal set of calibration coefficients

        :param np.ndarray mag_data: Numpy array of magnetic readings of shape (N,3)
        :param np.ndarray grav_data: Numpy array of gravity readings of shape (M,3)
        :return: (mag_accuracy, grav_accuracy) How well the calibrated model fits the data.
          Lower numbers are better
        """
        mag_accuracy = self.mag.fit_ellipsoid(mag_data)
        grav_accuracy = self.grav.fit_ellipsoid(grav_data)
        return mag_accuracy, grav_accuracy

    def fit_to_axis(self, data, axis="Y") -> float:
        """
        Take multiple reading with the device pointing in the same direction, but rotated around
        ``axis``. You can repeat this with several directions. This function will take
        this data and ensure that your sensors aligned relative to each other. This function
        requires that you have run `Calibration.fit_ellipsoid` beforehand.

        :param data: A list of paired magnetic and gravity readings e.g.:
          ``[(mag_data1, grav_data1), (mag_data2, grav_data2), ...]``, where ``mag_data1`` and
          ``grav_data1`` are (N,3) numpy arrays of readings around the axis in the first
          direction, and ``mag_data2`` and ``grav_data2`` are (M,3) numpy arrays of readings around
          the specified axis in another direction.
        :param axis: Axis you have rotated your device around. Defaults to ``"Y"``
        :return: Average standard deviation of readings in degrees, after calibration
        """
        mag_data = [x[0] for x in data]
        grav_data = [x[1] for x in data]
        self.mag.align_along_axis(mag_data, axis)
        self.grav.align_along_axis(grav_data, axis)
        return self.accuracy(data)

    def fit_non_linear(
        self, data, axis: str = "Y", param_count: int = 3, sensor: int = MAGNETOMETER
    ):
        """
        Compensate for devices which do not have a linear response between the
        magnetic or gravity field and their output. It is recommended to use a
        ``param_count`` of either 1, 3 or 5 for this function; odd numbers generally get better
        results, but there is a significant risk of overfitting if higher numbers are used.
        See `Underwood, Phil (2021) Non-linear Calibration of a Digital Compass and
        Accelerometer, Cave Radio Electronics Group Journal 114, pp7-10. June 2021
        <https://github.com/furbrain/SAP5/blob/master/doc/non-linear_calibration.pdf>`_
        for more details on the algorithm used. This function uses a Nelder-Mead minimisation
        process to find optimal values for each non-linear parameter.

        This function requires that you have run both `Calibration.fit_ellipsoid` and
        `Calibration.fit_to_axis` beforehand.

        :param data: A list of paired magnetic and gravity readings e.g.:
          ``[(mag_data1, grav_data1), (mag_data2, grav_data2), ...]``, where ``mag_data1`` and
          ``grav_data1`` is a (N,3) numpy array of readings around the axis in the first
          direction, and ``mag_data2`` and ``grav_data2`` is a (M,3) numpy array of readings around
          the specified axis in another direction.
        :param str axis: Axis along which the device has been rotated for the given readings.
        :param int param_count: Number of parameters to use per sensor axis. Larger numbers
          will take substantially more time to calculate. Default is 3
        :param sensor: Whether to calibrate the magnetometer, accelerometer or both. Must be one
          of ``Calibration.MAGNETOMETER``, ``Calibration.ACCELEROMETER``, or ``Calibration.BOTH``.
          Default is ``MAGNETOMETER``, as accelerometers are generally quite well-behaved.
        :return: Standard deviation of accuracy of calibration in degrees
        """
        params_per_sensor = param_count * 2
        if sensor == self.BOTH:
            params_total = params_per_sensor * 2
        else:
            params_total = params_per_sensor
        all_mag = np.concatenate(tuple(x[0] for x in data))
        all_grav = np.concatenate(tuple(x[1] for x in data))
        axis_index = "XYZ".index(axis.upper())  # which axis by number...

        # this is the minimisation function we wish to calculate...
        def min_func(x):
            if sensor & self.MAGNETOMETER:
                params = x[:params_per_sensor]
                final_params = np.zeros(param_count * 3)
                # do not optimize axis around which we are rotating - set these coefficients to zero
                if axis_index == 0:
                    final_params[:params_per_sensor] = params
                elif axis_index == 1:
                    final_params[:param_count] = params[:param_count]
                    final_params[-param_count:] = params[-param_count:]
                else:
                    final_params[-params_per_sensor:] = params[-params_per_sensor:]
                self.mag.set_non_linear_params(final_params)
            if sensor & self.ACCELEROMETER:
                params = x[-params_per_sensor:]
                final_params = np.zeros(param_count * 3)
                # do not optimize axis around which we are rotating - set these coefficients to zero
                # do not optimize axis around which we are rotating - set these coefficients to zero
                if axis_index == 0:
                    final_params[:params_per_sensor] = params
                elif axis_index == 1:
                    final_params[:param_count] = params[:param_count]
                    final_params[-param_count:] = params[-param_count:]
                else:
                    final_params[-params_per_sensor:] = params[-params_per_sensor:]
                self.grav.set_non_linear_params(final_params)
            return self.accuracy(data) + sum(self.uniformity(all_mag, all_grav))

        x_initial = np.full(params_total, 0.0)
        minimizer = nm.Minimizer(min_func, x_initial, adaptive=False)
        results = minimizer.optimize()
        if results["status"] != "success":
            raise RuntimeError("Unable to calibrate - optimisation routine failed")
        min_func(results["x"])
        return results["iterations"], self.accuracy(data)

    def fit_non_linear_quick(self, data, param_count: int = 3):
        # pylint: disable=invalid-name,too-many-locals
        """
        Compensate for devices which do not have a linear response between the
        magnetic or gravity field and their output. It is recommended to use a
        ``param_count`` of either 1, 3 or 5 for this function; odd numbers generally get better
        results, but there is a significant risk of overfitting if higher numbers are used.
        See `Underwood, Phil (2021) Non-linear Calibration of a Digital Compass and
        Accelerometer, Cave Radio Electronics Group Journal 114, pp7-10. June 2021
        <https://github.com/furbrain/SAP5/blob/master/doc/non-linear_calibration.pdf>`_
        for more details on the algorithm used. This function uses a least squares method to
        rapidly find a good set of parameters to use. It is *much* faster than
        `fit_non_linear`, but gives slightly less good results. Note it will
        only calibrate the magnetometer and is unable currently to apply a non-linear correction
        to the accelerometer. It will also only calibrate for rotations around the Y axis.

        This function requires that you have run both `Calibration.fit_ellipsoid` and
        `Calibration.fit_to_axis` beforehand.

        :param data: A list of paired magnetic and gravity readings e.g.:
          ``[(mag_data1, grav_data1), (mag_data2, grav_data2), ...]``, where ``mag_data1`` and
          ``grav_data1`` is a (N,3) numpy array of readings around the axis in the first
          direction, and ``mag_data2`` and ``grav_data2`` is a (M,3) numpy array of readings around
          the specified axis in another direction.
        :param int param_count: Number of parameters to use per sensor axis. Larger numbers
          will take substantially more time to calculate. Default is 3
        :return: Standard deviation of accuracy of calibration in degrees
        """
        self.mag.set_linear()
        self.grav.set_linear()
        expected_mags = []
        raw_mags = []
        for mag, grav in data:
            expected, raw = self._get_raw_and_expected_mag_data(mag, grav)
            expected_mags.extend(expected)
            raw_mags.extend(raw)
        # create least squares set of sums
        params = self._get_lstsq_non_linear_params(param_count, expected_mags, raw_mags)
        all_params = np.zeros(param_count * 3)
        all_params[:param_count] = params[:param_count]
        all_params[-param_count:] = params[-param_count:]
        self.mag.set_non_linear_params(all_params)
        return self.accuracy(data)

    def accuracy(self, data) -> float:
        """
        Calculate average accuracy for a set of multiple readings taken

        :param data: A list of paired magnetic and gravity readings e.g.:
          ``[(mag_data1, grav_data1), (mag_data2, grav_data2)]``, where ``mag_data1`` and
          ``grav_data1`` is a (N,3) numpy array of readings around the axis in the first
          direction, and ``mag_data2`` and ``grav_data2`` is a (M,3) numpy array of readings around
          the specified axis in another direction.
        :return: Average standard deviation of readings in degrees
        """
        results = 0
        for mag, grav in data:
            orientation = self.get_orientation_vector(mag, grav)
            stds = np.std(orientation, axis=0, ddof=1)
            results += np.linalg.norm(stds)
        return np.degrees(results / len(data))

    def uniformity(self, mag_data, grav_data):
        """
        Check the uniformity of the data - how well the calibrated data points fit on
        a sphere of radius 1.0

        :param np.ndarray mag_data: Numpy array of magnetic readings of shape (N,3)
        :param np.ndarray grav_data: Numpy array of gravity readings of shape (M,3)
        :return: (mag_accuracy, grav_accuracy) How well the calibrated model fits the data.
          Lower numbers are better
        """
        return self.mag.uniformity(mag_data), self.grav.uniformity(grav_data)

    @staticmethod
    def _get_lstsq_non_linear_params(param_count, expected_mags, raw_mags):
        """
        Create a set of non-linear parameters given a set of expected magnetometer readings
        and the actual magnetometer readings

        :param param_count: Number of parameters to use per sensor
        :param expected_mags: Expected readings
        :param raw_mags: Actual readings
        :return:
        """
        rbf = RBF(np.zeros(param_count))
        input_data = np.zeros((0, param_count * 2))
        output_data = np.zeros((0,))
        for raw, expected in zip(raw_mags, expected_mags):
            diff = expected - raw
            factors = rbf.get_gaussians(raw).transpose()
            temp_input = np.zeros((2, param_count * 2))
            temp_input[0, :param_count] = factors[0]
            temp_input[1, -param_count:] = factors[2]
            input_data = np.concatenate((input_data, temp_input))
            output_data = np.concatenate((output_data, np.array([diff[0], diff[2]])))
        params = solve_least_squares(input_data, output_data)
        return params

    def _get_raw_and_expected_mag_data(self, mag, grav):
        # pylint: disable=invalid-name,too-many-locals
        """
        Rotate a set of mag readings so that the effects of roll has been removed and calculate
        the average mag vector, then rotate this vector back to where it should be...
        :param grav:
        :param mag:
        :return:
        """
        rotated_mags = []
        rot_mats = []
        expected_mags = []
        raw_mags = []
        for m, g in zip(mag, grav):
            _, _, roll = self.get_angles(m, g)
            c = np.cos(np.radians(roll))
            s = np.sin(np.radians(roll))
            rot_mat = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            raw_mag = self.mag.apply(m)
            rotated_mag = np.dot(raw_mag.reshape((1, 3)), rot_mat.transpose())
            raw_mags.append(raw_mag)
            rotated_mags.append(rotated_mag.reshape((3,)))
            rot_mats.append(rot_mat)
        average_vector = np.mean(np.array(rotated_mags), axis=0).reshape((1, 3))
        for rot_mat in rot_mats:
            expected_mags.append(np.dot(average_vector, rot_mat).reshape((3,)))
        return expected_mags, np.array(raw_mags)

    def get_orientation_vector(self, mag, grav):
        """
        Given a set of magnetic readings and gravity readings, get the orientation
          of the device in world coordinates.

        :param np.array mag: Numpy array of shape(N,3) or (3,)
        :param np.array grav: Numpy array of shape(N,3) or (3,)
        :return: Device orientation in world coordinates.
        :rtype: Numpy array of same shape as mag, with device orientation.
        """
        orientation = self.get_orientation_matrix(mag, grav)
        if isinstance(orientation, list) and len(orientation[0].shape) == 2:
            return np.array([x[:, 1] for x in orientation])
        return orientation[:, 1]

    def get_orientation_matrix(self, mag, grav) -> np.ndarray:
        """
        Get the device orientation as an orthonormal matrix, given the magnetic and gravity readings

        :param numpy.ndarray mag: Magnetic readings, either as numpy array or sequence of 3
          floats
        :param numpy.ndarray grav: Gravity readings, either as numpy array or sequence of 3
          floats
        :return: Orthonormal matrix converting device coordinates to real world coordinates
        :rtype: numpy.ndarray
        """
        mag = normalise(self.mag.apply(mag))
        upward = normalise(self.grav.apply(grav)) * -1
        east = normalise(cross(mag, upward))
        north = normalise(cross(upward, east))
        if len(north.shape) > 1:
            orientation = [np.array([e, n, u]) for e, n, u in zip(east, north, upward)]
        else:
            orientation = np.array((east, north, upward))
        return orientation

    @staticmethod
    def matrix_to_angles(matrix: np.ndarray):
        """
        Extract the rotation angles from a matrix. Angles are "zxy" rotations (azimuth, pitch, roll)

        :param np.ndarray matrix:
        :return: azimuth, pitch, roll
        """
        if isinstance(matrix, list) and len(matrix[0].shape) == 2:
            m01 = _vector_from_matrices(matrix, 0, 1)
            m11 = _vector_from_matrices(matrix, 1, 1)
            m21 = _vector_from_matrices(matrix, 2, 1)
            m20 = _vector_from_matrices(matrix, 2, 0)
            m22 = _vector_from_matrices(matrix, 2, 2)
        else:
            m01 = matrix[0, 1]
            m11 = matrix[1, 1]
            m21 = matrix[2, 1]
            m20 = matrix[2, 0]
            m22 = matrix[2, 2]
        theta1 = np.arctan2(m01, m11)
        theta2 = np.arctan2(m21 * np.cos(theta1), m11)
        theta3 = np.arctan2(-m20, m22)
        if not isinstance(theta1, np.ndarray):
            azimuth = float(np.degrees(theta1)) % 360
        else:
            azimuth = np.array([x % 360 for x in np.degrees(theta1)])
        inclination = (
            (np.degrees(theta2) + 90) % 180
        ) - 90  # force to be in range -90,+90
        roll = np.degrees(theta3)
        return azimuth, inclination, roll

    @classmethod
    def angles_to_matrix(
        cls, azimuth: np.ndarray, inclination: np.ndarray, roll: np.ndarray
    ):
        # pylint: disable=invalid-name
        """
        Create a rotation matrix from angles "zxy" i.e azimmuth, inclination, roll. This is the
        invers of matrix_to_angles

        :param azimuth: float or np.array of floats for azimuth
        :param inclination: pitch(rotation around x axis)
        :param roll: roll around y axis
        :return:
        """
        if isinstance(azimuth, (float, int)) or len(azimuth) == 1:
            theta1 = np.radians(-azimuth)
            theta2 = np.radians(inclination)
            theta3 = np.radians(roll)
            c1 = np.cos(theta1)
            s1 = np.sin(theta1)
            c2 = np.cos(theta2)
            s2 = np.sin(theta2)
            c3 = np.cos(theta3)
            s3 = np.sin(theta3)

            matrix = np.array(
                [
                    [c1 * c3 - s1 * s2 * s3, -c2 * s1, c1 * s3 + c3 * s1 * s2],
                    [c3 * s1 + c1 * s2 * s3, c1 * c2, s1 * s3 - c1 * c3 * s2],
                    [-c2 * s3, s2, c2 * c3],
                ]
            )
            return matrix
        return [cls.angles_to_matrix(*args) for args in zip(azimuth, inclination, roll)]

    def find_similar_shots(
        self, mag: np.ndarray, grav: np.ndarray, precision=30, min_run=4
    ):
        """
        Find runs of shots that are within precision degrees of each other

        :param mag: numpy array of magnetic data
        :param grav: numpy array of accelerometer data
        :param precision: number of degrees shots should be within
        :param min_run: minimum length of run to find
        :return: list of start and finish indices for each run
        """
        angles = [self.get_angles(m, g) for m, g in zip(mag, grav)]
        azimuths, inclinations, _ = zip(*angles)
        azimuths = np.array(
            azimuths
        )  # convert to numpy array so we can do addition and modulo
        groups = []
        i = 0
        while i < len(azimuths) - min_run:
            for j in reversed(range(i + min_run, len(azimuths) + 1)):
                if self._is_a_run(azimuths[i:j], inclinations[i:j], precision):
                    groups.append([i, j])
                    i = j
                    break
            else:
                i += 1
        return groups

    @staticmethod
    def _is_a_run(azimuths: np.ndarray, inclinations: np.ndarray, precision: float):
        """
        Given a list of azimuths and inclinations, return true if they are all near each other
        :param azimuths: list of azimuths
        :param inclinations: list of inclinations
        :param precision: maximum allowed degrees of difference
        :return:
        """
        if max(inclinations) - min(inclinations) > precision:
            return False
        if max(azimuths) > 360 - precision:
            # rotate by 180 degs if shots near 359/0 degs
            azimuths = [(azimuth + 180) % 360 for azimuth in azimuths]
        if max(azimuths) - min(azimuths) > precision:
            return False
        return True

    def get_field_strengths(self, mag, grav):
        """
        Get field strength for magnetic and gravity components

        :param numpy.ndarray mag: Magnetic readings, either as numpy array or sequence of 3
          floats
        :param numpy.ndarray grav: Gravity readings, either as numpy array or sequence of 3
          floats
        :return: mag_field, grav_field
        :rtype: numpy.ndarrays if multiple readings given or floats
        """
        return self.mag.get_field_strength(mag), self.grav.get_field_strength(grav)

    def get_dips(self, mag, grav):
        """
        Get the magnetic field dip

        :param numpy.ndarray mag: Magnetic readings, either as numpy array or sequence of 3
          floats
        :param numpy.ndarray grav: Gravity readings, either as numpy array or sequence of 3
          floats
        :return: dip angle(s) in degrees
        :rtype: numpy.ndarrays if multiple readings given or floats
        """
        normalised_mags = normalise(self.mag.apply(mag))
        normalised_gravs = normalise(self.grav.apply(grav))
        if len(normalised_gravs.shape) > 1:
            dot_products = [
                np.dot(m, g) for m, g in zip(normalised_mags, normalised_gravs)
            ]
            dot_products = np.array(dot_products)
        else:
            dot_products = np.dot(normalised_mags, normalised_gravs)
        dips = 90 - np.degrees(arccos(dot_products))
        return dips

    def set_expected_mean_dip(self, mag, grav):
        """
        Store an expected dip and standard deviations. This will be used for
        magnetic and (gravitational!!) anomaly detection

        :param numpy.ndarray mag: Magnetic readings, either as numpy array or sequence of 3
          floats
        :param numpy.ndarray grav: Gravity readings, either as numpy array or sequence of 3
          floats
        """
        dips = self.get_dips(mag, grav)
        self.dip_avg = np.mean(dips)

    def set_field_characteristics(self, mag, grav):
        """
        Store magnetic and gravity field strengths and also dip angles. This will be used for
        magnetic and (gravitational!!) anomaly detection

        :param numpy.ndarray mag: Magnetic readings, either as numpy array or sequence of 3
          floats
        :param numpy.ndarray grav: Gravity readings, either as numpy array or sequence of 3
          floats
        :return:
        """
        self.mag.set_expected_field_strengths(mag)
        self.grav.set_expected_field_strengths(grav)
        self.set_expected_mean_dip(mag, grav)

    def raise_if_anomaly(self, mag, grav, strictness: Strictness = _DEFAULT_STRICTNESS):
        """
        Raises an error if magnetic field strength and dip are not similar to during calibration

        :param numpy.ndarray mag: Magnetic readings, sequence of 3 floats
        :param numpy.ndarray grav: Gravity readings, sequence of 3 floats
        :param Strictness strictness: NamedTuple containing the following entries

          * ``mag``: Acceptable percentage difference in magnetic field strength
          * ``grav``: Acceptable percentage difference in gravity field strength
          * ``dip``: Acceptable difference in dip in degrees

          If not specified will default to 2% for ``mag`` and ``grav`` and 3° for ``dip``
        :return: None
        :raises:
          * ``MagneticAnomalyError`` if magnetic field strength too big or small.
          * ``GravityAnomalyError`` if gravity field strength too big or small - usually
            occurs if movement during read
          * ``DipAnomalyError`` if magnetic field dip too big or small
        """
        try:
            self.mag.raise_if_anomaly(mag, strictness.mag / 100)
        except SensorError as exc:
            raise MagneticAnomalyError(exc.args) from exc
        try:
            self.grav.raise_if_anomaly(grav, strictness.grav / 100)
        except SensorError as exc:
            raise GravityAnomalyError(exc.args) from exc
        dip = self.get_dips(mag, grav)
        acceptable = strictness.dip
        variation = abs(self.dip_avg - dip)
        if variation > acceptable:
            raise DipAnomalyError(
                f"Magnetic dip {dip} out of limits {self.dip_avg-acceptable} - "
                + f"{self.dip_avg+acceptable}"
            )
