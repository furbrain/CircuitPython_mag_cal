# SPDX-FileCopyrightText: 2017 Scott Shawcroft, written for Adafruit Industries
# SPDX-FileCopyrightText: Copyright (c) 2022 Phil Underwood for Underwood Underground
#
# SPDX-License-Identifier: Unlicense
import time

from mag_cal.calibration import Calibration
from mag_cal.utils import read_fixture

PATH = "../tests/fixtures/cal_data/hj.json"

with open(PATH) as f:
    aligned, mag, grav = read_fixture(f.read())
calib = Calibration()
start = time.time()
print(f"Starting at:\t{time.time()-start}")
calib.fit_ellipsoid(mag, grav)
print(f"Ellipsoid at:\t{time.time()-start}")
calib.align_along_axis(aligned)
print(f"Aligned at:\t{time.time()-start}")
calib.apply_non_linear_correction_quick(aligned, param_count=5)
print(f"Quick NL at:\t{time.time()-start}")
calib.apply_non_linear_correction(aligned, param_count=5)
print(f"Slow NL at:\t{time.time()-start}")
