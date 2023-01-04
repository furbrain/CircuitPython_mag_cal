# SPDX-FileCopyrightText: Copyright (c) 2022 Phil Underwood for Underwood Underground
#
# SPDX-License-Identifier: MIT
"""
Calibration
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
from .calibration import Calibration
from .axes import Axes
from .sensor import Sensor
from .utils import NotCalibrated

__all__ = ["Calibration", "Axes", "Sensor", "NotCalibrated"]
