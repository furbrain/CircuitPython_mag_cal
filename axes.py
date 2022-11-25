# SPDX-FileCopyrightText: Copyright (c) 2022 Phil Underwood for Underwood Underground
#
# SPDX-License-Identifier: MIT
"""
Switch between orthogonal coordinate systems
"""

import numpy as np


class Axes:
    """
    This represents the orientation of a sensor in respect to the device orientation
    """

    def __init__(self, axes):
        """
        :param str axes: A string representing how your sensor is mounted with respect to your
          device. For each axis XYZ of your device (see `mag_cal` for axis descriptions), state the
          corresponding axis of your sensor. Add a ``+`` or ``-`` in front to let us know if it is
          inverted. So for a sensor that is correctly mounted it will be ``"+X+Y+Z"``. If the
          sensors Y axis points to the left of the device, the X is forwards and Z is down, specify
          this as ``"-Y+X+Z"``
        """
        self.indices = [0, 0, 0]
        self.polarities = [0, 0, 0]
        if len(axes) != 6:
            raise ValueError("Axes must be 6 characters long")
        for char in "XYZ":
            if char not in axes.upper():
                raise ValueError(f"Axis {char} must be present in axis string")
        for i in range(3):
            axis, polarity = self._get_axis_and_polarity(axes[i * 2 : i * 2 + 2])
            self.indices[i] = axis
            self.polarities[i] = polarity

    @staticmethod
    def _get_axis_and_polarity(fragment: str):
        if len(fragment) != 2:
            raise ValueError("Axis descriptor must have two characters")
        fragment = fragment.upper()
        if fragment[0] == "-":
            polarity = -1
        elif fragment[0] == "+":
            polarity = +1
        else:
            raise ValueError("Axis prefix must be '+' or '-'")
        if fragment[1] == "X":
            axis = 0
        elif fragment[1] == "Y":
            axis = 1
        elif fragment[1] == "Z":
            axis = 2
        else:
            raise ValueError("Axis must be X,Y or Z")
        return axis, polarity

    def fix_axes(self, data: np.ndarray):
        """
        Transform raw ``data`` from sensor coordinates to raw device coordinates
        :param data: raw sensor data as a np.array((...,3))
        :return: np.array with same dimensions as ``data``
        """
        new_data = np.zeros_like(data)
        data = np.array(data)
        for i in range(3):
            new_data[..., i] = data[..., self.indices[i]] * self.polarities[i]
        return new_data