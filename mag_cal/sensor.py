# SPDX-FileCopyrightText: Copyright (c) 2022 Phil Underwood for Underwood Underground
#
# SPDX-License-Identifier: MIT
"""
Provides the `Sensor` class that represents a a single sensor and its calibration
"""
try:
    from typing import List
except ImportError:
    pass

try:
    import numpy as np
except ImportError:
    from ulab import numpy as np

from .axes import Axes
from .utils import solve_least_squares, normalise, NotCalibrated, cross
from .rbf import RBF


class Sensor:
    """
    This represents the calibration coefficients for a single sensor
    """

    def __init__(self, axes="+X+Y+Z"):
        """
        Create a sensor object, with axes set

        :param str axes: A string representing how your sensor is mounted with respect to your
          device. For each axis XYZ of your device (see above for axis descriptions), state the
          corresponding axis of your sensor. Add a ``+`` or ``-`` to let us know if it is inverted.
          So for a sensor that is correctly mounted it will be ``"+X+Y+Z"``. If the sensors Y axis
          points to the left of the device, the X is forwards and Z is down, specify this as
          ``"-Y+X+Z"``. Default is ``+X+Y+Z``
        """
        self.axes = Axes(axes)
        self.transform: np.ndarray = None
        self.centre: np.ndarray = None
        self.rbfs: List[RBF] = []

    def fit_ellipsoid(self, data: np.ndarray) -> float:
        # pylint: disable=too-many-locals, invalid-name
        """
        Take multiple sets of readings in various directions. You can then use this function
        to determine an ideal set of calibration coefficients.

        :param np.ndarray data: Numpy array of readings of shape (N,3)
        :return: accuracy - How well the calibrated model fits the data; lower numbers are better.
        :rtype: float
        """
        fixed_data = self.axes.fix_axes(data)
        x = fixed_data[:, 0:1]
        y = fixed_data[:, 1:2]
        z = fixed_data[:, 2:3]
        output_array = np.ones(x.shape)

        input_array = np.concatenate(
            (x * x, y * y, z * z, 2 * x * y, 2 * x * z, 2 * y * z, 2 * x, 2 * y, 2 * z),
            axis=1,
        )
        coeff = solve_least_squares(input_array, output_array)
        a, b, c, d, e, f, g, h, i = coeff.flatten()
        A4 = np.array([[a, d, e, g], [d, b, f, h], [e, f, c, i], [g, h, i, -1]])
        A3 = A4[0:3, 0:3]
        vghi = np.array([-g, -h, -i])
        self.centre = solve_least_squares(A3, vghi)
        T = np.eye(4)
        T[3, 0:3] = self.centre
        B4 = np.dot(np.dot(T, A4), T.transpose())
        B3 = B4[0:3, 0:3] / -B4[3, 3]
        e, v = np.linalg.eig(B3)
        self.transform = np.dot(np.dot(v, np.sqrt(np.diag(e))), v.transpose())
        return self.uniformity(data)

    def _align_to_vector(self, vector, axis):
        """
        Rotate transformation matrix, such that any points along ``vector`` are now on the
        specified ``axis``.

        :param np.ndarray vector: numpy array of shape (3) to align with
        :param np.ndarray axis: Axis to align to, must be one of "X", "Y", or "Z".
        """
        if axis == "X":
            vector_x = vector
            vector_z = cross(vector_x, np.array((0, 1, 0)))
            vector_y = cross(vector_z, vector_x)
        elif axis == "Y":
            vector_y = vector
            vector_x = cross(vector_y, np.array((0, 0, 1)))
            vector_z = cross(vector_x, vector_y)
        elif axis == "Z":
            vector_z = vector
            vector_y = cross(vector_z, np.array((1, 0, 0)))
            vector_x = cross(vector_y, vector_z)
        else:
            raise ValueError("Axis must be X, Y or Z")
        vector_x = normalise(vector_x)
        vector_y = normalise(vector_y)
        vector_z = normalise(vector_z)
        mat = np.concatenate((vector_x, vector_y, vector_z)).reshape((3, 3))
        self.transform = np.dot(mat.transpose(), self.transform)

    def align_along_axis(self, data, axis="Y"):
        """
        Calibrate for any mechanical placement error.

        :param List[np.ndarray] data: A list of sets of data - numpy array of shape (N,3). Each
          set should have been taken with the device pointing at a fixed target, but rotated through
          different angles
        :param str axis: Axis that the device has been rotated around, must be one of "X", "Y",
          or "Z". Default is "Y".
        :return: None
        """

        result = np.zeros(3)
        if axis not in "XYZ":
            raise ValueError('Axis must be one of "X", "Y", or "Z"')
        axis_index = "XYZ".index(axis)
        for points in data:
            vector = self._find_plane(points)
            if vector[axis_index] < 0:
                vector *= -1
            result += vector
        self._align_to_vector(result, axis)

    def _find_plane(self, data):
        """
        Find a plane that fits the data points given

        :param np.ndarray data: set of data of shape (N,3)
        :return: Numpy array with the vector corresponding to the plane that best fits the data
        """
        data = self.apply(data)
        output = np.ones(data.shape[:1])
        result = solve_least_squares(data, output)
        return normalise(result)

    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Take a set of raw data and apply the calibration to it

        :param np.ndarray data: Numpy array of shape (3,) or (N,3)
        :return: numpy array with same shape as ``data``
        """
        if self.transform is None:
            raise NotCalibrated("This sensor is not yet calibrated")
        data = self.axes.fix_axes(data)
        data -= self.centre
        if self.rbfs:
            data = self._apply_non_linear(data)
        if len(data.shape) == 1:
            data = np.dot(data.reshape((1, 3)), self.transform)[0]
        else:
            data = np.dot(data, self.transform)
        return data

    def set_non_linear_params(self, params: np.ndarray):
        """
        Set the parameters for the radial basis functions

        :param np.ndarray params: Numpy matrix or vector with N*3 elements, where N is the number
          of parameters per axis
        """
        params = np.array(params)
        params = params.reshape((3, params.size // 3))
        self.rbfs = [RBF(p) for p in params]

    def set_linear(self):
        """
        Disable non-linear adjustments
        """
        self.rbfs = []

    def _apply_non_linear(self, vectors: np.ndarray):
        # pylint: disable=unsubscriptable-object
        vectors = np.array(vectors)
        scale = self.transform[0, 0]
        normalised_v = vectors * scale
        if len(vectors.shape) == 2:
            vectors[:, 0] += self.rbfs[0](normalised_v[:, 0]) / scale
            vectors[:, 1] += self.rbfs[1](normalised_v[:, 1]) / scale
            vectors[:, 2] += self.rbfs[2](normalised_v[:, 2]) / scale
        else:
            vectors[0] += self.rbfs[0](normalised_v[0]) / scale
            vectors[1] += self.rbfs[1](normalised_v[1]) / scale
            vectors[2] += self.rbfs[2](normalised_v[2]) / scale
        return vectors

    def uniformity(self, data: np.ndarray):
        """
        Check the uniformity of the data points after calibration. This is measured as
        the standard deviation of the absolute magnitude of each measurement

        :param np.ndarray data: Numpy array of shape (N,3)
        :return: Calculated uniformity as above
        """
        # get the radii of of all the data points
        radii = np.linalg.norm(self.apply(data), axis=1)
        # get the standard deviation of these values (mean radius should be 1)
        return np.sqrt(np.mean((radii - 1) ** 2))

    def as_dict(self):
        """
        Return the calibration parameters as a dict, can be reloaded using `Sensor.from_dict`

        :return: Dict containing all the calibration data for this sensor
        """
        results = {
            "axes": str(self.axes),
            "transform": self.transform.tolist(),
            "centre": self.centre.tolist(),
            "rbfs": [x.as_list() for x in self.rbfs],
        }
        return results

    @classmethod
    def from_dict(cls, dct: dict) -> "Sensor":
        """
        Create a new `Sensor` instance from the given dict, which should have been created
        using `as_dict`.

        :param dict dct: Dict of values as created by `as_dict`
        :return: New Sensor object, initialised with given data
        """
        instance = cls(dct["axes"])
        instance.transform = np.array(dct["transform"])
        instance.centre = np.array(dct["centre"])
        instance.rbfs = [RBF(x) for x in dct["rbfs"]]
        return instance
