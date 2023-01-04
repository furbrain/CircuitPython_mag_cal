# SPDX-FileCopyrightText: Copyright (c) 2022 Phil Underwood for Underwood Underground
#
# SPDX-License-Identifier: Unlicense
from mag_cal.calibration import Calibration
from mag_cal.utils import read_fixture

PATH = "../tests/fixtures/cal_data/hj2.json"

with open(PATH) as f:
    aligned, grav, mag = read_fixture(f.read())

calib = Calibration()
calib.fit_ellipsoid(mag, grav)
calib.fit_to_axis(aligned)
calib.fit_non_linear_quick(aligned, param_count=5)

# calib.fit_non_linear(aligned, param_count=3)
for m, g in zip(mag, grav):
    azimuth, inclination, roll = calib.get_angles(m, g)
    print(f"{azimuth:05.1f}° {inclination:+05.1f}° {roll:+04.0f}°")
