# SPDX-FileCopyrightText: Copyright (c) 2022 Phil Underwood for Underwood Underground
#
# SPDX-License-Identifier: MIT
from pathlib import Path
from unittest import TestCase

import numpy as np

from mag_cal.sensor import Sensor
from mag_cal.utils import read_fixture


class TestSensor(TestCase):
    def setUp(self) -> None:
        self.fixtures = {}
        for fname in (Path(__file__).parent / "fixtures" / "cal_data").glob("*.json"):
            with open(fname) as fixture:
                _, grav, mag = read_fixture(fixture.read())
                self.fixtures[fname] = (mag, grav)

    def test_fit_ellipsoid(self):
        for _, (mag, grav) in self.fixtures.items():
            for data in (mag, grav):
                calib = Sensor()
                uniformity = calib.fit_ellipsoid(data)
                self.assertGreater(0.01, uniformity)
                new_data = calib.apply(data)
                norms = np.linalg.norm(new_data, axis=-1)
                self.assertGreater(1.02, norms.max())
                self.assertLess(0.98, norms.min())

    def test_align_along_axis(self):
        # pylint: disable=protected-access
        non_axis_data = []
        for mag, grav in self.fixtures.values():
            for data in (mag, grav):
                calib = Sensor()
                accuracy_before = calib.fit_ellipsoid(data)
                calib.align_along_axis([data[8:16], data[16:24]])
                accuracy_after = calib.uniformity(data)
                # make sure uniformity not significantly impaired by alignment
                self.assertGreater(accuracy_before + 0.001, accuracy_after)
                aligned_axis = calib._find_plane(
                    data[8:16]
                )  # aligned axis should be [0 1 0]
                non_axis_data.append(aligned_axis[0])
                non_axis_data.append(aligned_axis[2])
                aligned_axis = calib._find_plane(data[16:24])
                non_axis_data.append(aligned_axis[0])
                non_axis_data.append(aligned_axis[2])
        self.assertLess(
            max(non_axis_data), 0.005
        )  # equivalent to less than 0.28 degrees error
