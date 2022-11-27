# SPDX-FileCopyrightText: Copyright (c) 2022 Phil Underwood for Underwood Underground
#
# SPDX-License-Identifier: MIT
import json
import math
from pathlib import Path
from unittest import TestCase

import numpy as np

from sensor import Sensor


class TestSensor(TestCase):
    def setUp(self) -> None:
        self.fixtures = {}
        for fname in (Path(__file__).parent / "fixtures" / "cal_data").glob("*.json"):
            with open(fname) as fixture:
                data = json.load(fixture)
                mag = data["shots"]["mag"]
                grav = data["shots"]["grav"]
                mag = [x for x in mag if not math.isnan(x)]
                mag = np.reshape(mag, (-1, 3))
                grav = [x for x in grav if not math.isnan(x)]
                grav = np.reshape(grav, (-1, 3))
                self.fixtures[fname] = (mag, grav)

    def test_fit_ellipsoid(self):
        for mag, grav in self.fixtures.values():
            for data in (mag, grav):
                calib = Sensor()
                calib.fit_ellipsoid(data)

                new_data = calib.apply(data)
                norms = np.linalg.norm(new_data, axis=-1)
                self.assertGreater(1.02, norms.max())
                self.assertLess(0.98, norms.min())

    def test_align_laser(self):
        non_axis_data = []
        for mag, grav in self.fixtures.values():
            for data in (mag, grav):
                calib = Sensor()
                calib.fit_ellipsoid(data)
                calib.align_along_axis([data[8:16], data[16:24]])
                aligned_axis = calib.find_plane(data[8:16])
                non_axis_data.append(aligned_axis[0])
                non_axis_data.append(aligned_axis[2])
                aligned_axis = calib.find_plane(data[16:24])
                non_axis_data.append(aligned_axis[0])
                non_axis_data.append(aligned_axis[2])
        self.assertLess(max(non_axis_data), 0.005)  # equivalent to less than
