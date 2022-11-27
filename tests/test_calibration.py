# SPDX-FileCopyrightText: Copyright (c) 2022 Phil Underwood for Underwood Underground
#
# SPDX-License-Identifier: MIT
import json
import math
from pathlib import Path
from unittest import TestCase

import numpy as np

from calibration import Calibration


class TestCalibration(TestCase):
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
            calib = Calibration()
            mag_accuracy, grav_accuracy = calib.fit_ellipsoid(mag, grav)
            self.assertGreater(0.01, mag_accuracy)
            self.assertGreater(0.001, grav_accuracy)
