# SPDX-FileCopyrightText: Copyright (c) 2022 Phil Underwood for Underwood Underground
#
# SPDX-License-Identifier: MIT
"""
`mag_cal`
================================================================================

Calibrate magnetometer and accelerometer readings


* Author(s): Phil Underwood

Implementation Notes
--------------------

**Software and Dependencies:**

* Adafruit CircuitPython firmware for the supported boards: https://circuitpython.org/downloads


Axes
----

There are several axis conventions used.

* World coordinates: this represents the "real world", X is due east, Y is due North, Z is Up
* Device coordinates: this represents the device. Y is the primary axis - the direction of
  travel of direction or direction of a pointer. Z is up, and X is to the right if Y is
  facing away from you and Z is up.

"""
from axes import Axes
from rbf import RBF

try:
    from typing import Tuple, List, Optional
except ImportError:
    # ignore if running in CircuitPython/MicroPython
    pass

try:
    # work with normal numpy
    import numpy as np
    import scipy as spy

    ULAB_PRESENT = False
except ImportError:
    # or work with ulab
    from ulab import numpy as np
    from ulab import scipy as spy
    import lstsq

    ULAB_PRESENT = True

__version__ = "0.0.0+auto.0"
__repo__ = "https://github.com/furbrain/CircuitPython_mag_cal.git"


def solve_least_squares(A: np.ndarray, B: np.ndarray):
    # pylint: disable=invalid-name
    """
    Calculate x such that Ax=B
    :param A:
    :param B:
    :return: x
    """
    if ULAB_PRESENT:
        # pylint: disable=used-before-assignment
        coeffs, _ = lstsq.lstsq(A, B)
    else:
        coeffs, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    return coeffs


class NotCalibrated(Exception):
    """
    Error to raise when calibration has not been done yet
    """


class Transform:
    """
    This represents the calibration coefficients for a single sensor
    """

    def __init__(self, axes):
        """
        Create a transform object, with axes set
        :param str axes: A string representing how your sensor is mounted with respect to your
          device. For each axis XYZ of your device (see above for axis descriptions), state the
          corresponding axis of your sensor. Add a ``+`` or ``-`` to let us know if it is inverted.
          So for a sensor that is correctly mounted it will be ``"+X+Y+Z"``. If the sensors Y axis
          points to the left of the device, the X is forwards and Z is down, specify this as
          ``"-Y+X+Z"``
        """
        self.axes = Axes(axes)
        self.non_linear = 0
        self.transform = None
        self.centre = None
        self.rbfs: Optional[List[RBF]] = None

    def fit_ellipsoid(self, data: np.ndarray) -> float:
        # pylint: disable=too-many-locals, invalid-name
        """
        Take multiple sets of readings in various directions. You can then use this function
        to determine an ideal set of calibration coefficients

        :param np.ndarray data: Numpy array of readings of shape (dims,3)
        :return: accuracy - How well the calibrated model fits the data lower numbers are better.
        :rtype: float
        """
        data = self.axes.fix_axes(data)
        x = data[:, 0:1]
        y = data[:, 1:2]
        z = data[:, 2:3]
        output_array = np.ones_like(x)

        input_array = np.concatenate(
            (x * x, y * y, z * z, 2 * x * y, 2 * x * z, 2 * y * z, 2 * x, 2 * y, 2 * z),
            axis=1,
        )
        coeff = solve_least_squares(input_array.transpose(), output_array)
        a, b, c, d, e, f, g, h, i = coeff
        A4 = np.array([[a, d, e, g], [d, b, f, h], [e, f, c, i], [g, h, i, -1]])
        A3 = A4[0:3, 0:3]
        vghi = np.array([-g, -h, -i])
        self.centre = solve_least_squares(A3, vghi)
        T = np.identity(4)
        T[3, 0:3] = self.centre
        B4 = np.dot(np.dot(T, A4), T.transpose())
        B3 = B4[0:3, 0:3] / -B4[3, 3]
        e, v = np.linalg.eig(B3)
        self.transform = np.dot(v, np.sqrt(np.diag(e))).dot(v.transpose())

    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Take a set of raw data and apply the calibration to it
        :param np.ndarray data: Numpy array of shape (3,) or (dims,3)
        :return: numpy array with same shape as ``data``
        """
        if self.transform is None:
            raise NotCalibrated("This sensor is not yet calibrated")
        data = self.axes.fix_axes(data)
        data -= self.centre
        if self.non_linear != 0:
            data = self._apply_non_linear(data)
        data = np.dot(data, self.transform)
        return data

    def _apply_non_linear(self, vectors):
        # pylint: disable=unsubscriptable-object
        vectors = np.array(vectors)
        scale = self.transform[0, 0]
        normalised_v = vectors * scale
        vectors[..., 0] += self.rbfs[0](normalised_v[..., 0]) / scale
        vectors[..., 1] += self.rbfs[1](normalised_v[..., 1]) / scale
        vectors[..., 2] += self.rbfs[2](normalised_v[..., 2]) / scale
        return vectors


class Calibration:
    """
    An object representing the calibration coefficients for a combined
        magnetometer and accelerometer
    """

    def __init__(self, mag_axes: str = "+X+Y+Z", grav_axes: str = None):
        """
        Create an object representing the calibration coefficients for a combined
        magnetometer and accelerometer

        :param str mag_axes: A string representing how your magnetometer is mounted with respect to
          your device. For each axis XYZ of your device (see above for axis descriptions), state the
          corresponding axis of your sensor. Add a ``+`` or ``-`` to let us know if it is inverted.
          So for a sensor that is correctly mounted it will be ``"+X+Y+Z"``. If the sensors Y axis
          points to the left of the device, the X is forwards and Z is down, specify this as
          ``"-Y+X+Z"``
        :param str grav_axes: Same format as ``mag_axes`` but for the accelerometer. Default is copy
          of mag_axes.
        """
        if grav_axes is None:
            grav_axes = mag_axes
        self.non_linear = 0
        self.mag = Transform(axes=mag_axes)
        self.grav = Transform(axes=grav_axes)
        self.ready = False

    def to_json(self) -> str:
        """
        Report current calibration as JSON text

        :return: The calibration in JSON format
        :rtype: str
        """

    def from_json(self, text: str) -> None:
        """
        Read some JSON text previosuly produced by _`to_json`

        :param text: JSON text to read
        :return: None
        """

    def fit_ellipsoid(
        self, mag_data: np.ndarray, grav_data: np.ndarray
    ) -> Tuple[float, float]:
        """
        Take multiple sets of readings in various directions. You can then use this function
        to determine an ideal set of calibration coefficients

        :param np.ndarray mag_data: Numpy array of magnetic readings of shape (dims,3)
        :param np.ndarray grav_data: Numpy array of gravity readings of shape (M,3)
        :return: (mag_accuracy, grav_accuracy) How well the calibrated model fits the data.
          Lower numbers are better
        """

    def align_axes(self, data, axis="Y", non_linear=0) -> float:
        """
        Take multiple reading with the device pointing in the same direction, but rotated around
        ``axis``. You can repeat this with several different directions. This function will take
        this data and ensure that your sensors aligned relative to each other. It will also apply a
        non-linear correction

        :param data: A list of paired magnetic and gravity readings e.g.:
          ``[(mag_data1, grav_data1), (mag_data2, grav_data2)]``, where ``mag_data1`` and
          ``grav_data1`` is a (dims,3) numpy array of readings around the axis in the first
          direction, and ``mag_data2`` and ``grav_data2`` is a (M,3) numpy array of readings around
          the specified axis in another direction.
        :param axis: Axis you have rotated your device around. Defaults to ``"Y"``
        :param int non_linear: Set to non-zero if you want to use a non-linear correction. This
          allows you to compensate for devices which do not have a linear response between the
          magnetic or gravity field and their output. It is recommended to use either 1,3 or 5 for
          this function; odd numbers generally get better results, but there is a significant risk
          of overfitting if higher numbers are used.
          See `Underwood, Phil (2021) Non-linear Calibration of a Digital Compass and
          Accelerometer, Cave Radio Electronics Group Journal 114, pp7-10. June 2021
          <https://github.com/furbrain/SAP5/blob/master/doc/non-linear_calibration.pdf>`_
          for more details on the algorithm used.
        :return: Standard deviation of accuracy of calibration in degrees
        """
        ...

    def get_orientation_matrix(self, mag_data, grav_data) -> np.ndarray:
        """
        Get the device orientation as an orthonormal matrix, given the magnetic and gravity readings

        :param numpy.ndarray mag_data: Magnetic readings, either as numpy array or sequence of 3
        floats
        :param numpy.ndarray grav_data: Gravity readings, either as numpy array or sequence of 3
        floats
        :return: Orthonormal matrix converting device coordinates to real world coordinates
        :rtype: numpy.ndarray
        """
        ...

    def get_angles(self, mag_data, grav_data) -> Tuple[float, float]:
        """
        Get device azimuth(bearing) and inclination, given the magnetic and gravity readings

        :param np.ndarray mag_data: Magnetic readings, either as numpy array or sequence of 3 floats
        :param np.ndarray grav_data: Gravity readings, either as numpy array or sequence of 3 floats
        :return: (azimuth, inclination) in degrees
        """
        ...
