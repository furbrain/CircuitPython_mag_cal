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
from sensor import Sensor

try:
    from typing import Tuple
except ImportError:
    # ignore if running in CircuitPython/MicroPython
    pass

try:
    # work with normal numpy
    import numpy as np

    # import scipy as spy

except ImportError:
    # or work with ulab
    from ulab import numpy as np

    # from ulab import scipy as spy


__version__ = "0.0.0+auto.0"
__repo__ = "https://github.com/furbrain/CircuitPython_mag_cal.git"


class Calibration:
    """
    Object representing a magnetometer and accelerometer calibration
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
          ``"-Y+X-Z"``
        :param str grav_axes: Same format as ``mag_axes`` but for the accelerometer. Default is copy
          of mag_axes.
        """
        if grav_axes is None:
            grav_axes = mag_axes
        self.mag = Sensor(axes=mag_axes)
        self.grav = Sensor(axes=grav_axes)
        self.ready = False

    def to_json(self) -> str:
        """
        Report current calibration as JSON text

        :return: The calibration in JSON format
        :rtype: str
        """

    # def from_json(self, text: str) -> None:
    #     """
    #     Read some JSON text previosuly produced by _`to_json`
    #
    #     :param text: JSON text to read
    #     :return: None
    #     """
    #
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


# def align_axes(self, data, axis="Y", non_linear=0) -> float:
#     """
#     Take multiple reading with the device pointing in the same direction, but rotated around
#     ``axis``. You can repeat this with several different directions. This function will take
#     this data and ensure that your sensors aligned relative to each other. It will also apply a
#     non-linear correction
#
#     :param data: A list of paired magnetic and gravity readings e.g.:
#       ``[(mag_data1, grav_data1), (mag_data2, grav_data2)]``, where ``mag_data1`` and
#       ``grav_data1`` is a (N,3) numpy array of readings around the axis in the first
#       direction, and ``mag_data2`` and ``grav_data2`` is a (M,3) numpy array of readings around
#       the specified axis in another direction.
#     :param axis: Axis you have rotated your device around. Defaults to ``"Y"``
#     :param int non_linear: Set to non-zero if you want to use a non-linear correction. This
#       allows you to compensate for devices which do not have a linear response between the
#       magnetic or gravity field and their output. It is recommended to use either 1,3 or 5 for
#       this function; odd numbers generally get better results, but there is a significant risk
#       of overfitting if higher numbers are used.
#       See `Underwood, Phil (2021) Non-linear Calibration of a Digital Compass and
#       Accelerometer, Cave Radio Electronics Group Journal 114, pp7-10. June 2021
#       <https://github.com/furbrain/SAP5/blob/master/doc/non-linear_calibration.pdf>`_
#       for more details on the algorithm used.
#     :return: Standard deviation of accuracy of calibration in degrees
#     """
#     ...
#
# def get_orientation_matrix(self, mag_data, grav_data) -> np.ndarray:
#     """
#     Get the device orientation as an orthonormal matrix, given the magnetic and gravity readings
#
#     :param numpy.ndarray mag_data: Magnetic readings, either as numpy array or sequence of 3
#       floats
#     :param numpy.ndarray grav_data: Gravity readings, either as numpy array or sequence of 3
#       floats
#     :return: Orthonormal matrix converting device coordinates to real world coordinates
#     :rtype: numpy.ndarray
#     """
#     ...
#
# def get_angles(self, mag_data, grav_data) -> Tuple[float, float]:
#     """
#     Get device azimuth(bearing) and inclination, given the magnetic and gravity readings
#
#     :param np.ndarray mag_data: Magnetic readings, either as numpy array or sequence of 3 floats
#     :param np.ndarray grav_data: Gravity readings, either as numpy array or sequence of 3 floats
#     :return: (azimuth, inclination) in degrees
#     """
#     ...
