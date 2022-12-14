# SPDX-FileCopyrightText: Copyright (c) 2022 Phil Underwood for Underwood Underground
#
# SPDX-License-Identifier: MIT
import itertools
import json
import math
import unittest
from pathlib import Path
from unittest import TestCase

import numpy as np

from calibration import Calibration


class TestCalibration(TestCase):
    def setUp(self) -> None:
        np.set_printoptions(precision=4, suppress=True)
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
                mag_aligned = [mag[8:16], mag[16:24]]
                grav_aligned = [grav[8:16], grav[16:24]]
                aligned_data = list(zip(mag_aligned, grav_aligned))
                self.fixtures[fname] = (mag, grav, aligned_data)

    def test_fit_ellipsoid(self):
        for mag, grav, _ in self.fixtures.values():
            calib = Calibration()
            mag_accuracy, grav_accuracy = calib.fit_ellipsoid(mag, grav)
            self.assertGreater(0.01, mag_accuracy)
            self.assertGreater(0.001, grav_accuracy)

    def test_align_along_axis(self):
        for mag, grav, aligned_data in self.fixtures.values():
            calib = Calibration()
            calib.fit_ellipsoid(mag, grav)
            before = calib.accuracy(aligned_data)
            calib.align_along_axis(aligned_data, axis="Y")
            after = calib.accuracy(aligned_data)
            print(before, after)
            # we should expect accuracy to at least double after alignment
            self.assertLess(2.0, before / after)

    def test_non_linear(self):
        total_non_linear = 0
        total_axis = 0
        for mag, grav, aligned_data in self.fixtures.values():
            calib = Calibration()
            calib.fit_ellipsoid(mag, grav)
            calib.accuracy(aligned_data)
            axis = calib.align_along_axis(aligned_data, axis="Y")
            _, non_linear = calib.apply_non_linear_correction(
                aligned_data, sensor=Calibration.MAGNETOMETER, param_count=3
            )
            total_non_linear += non_linear
            total_axis += axis
            print(non_linear, axis)
            self.assertLess(
                non_linear, axis
            )  # check non_linear process improves accuracy!
        print(total_non_linear / total_axis)

    def test_non_linear_quick(self):
        total_non_linear = 0
        total_axis = 0
        for mag, grav, aligned_data in self.fixtures.values():
            calib = Calibration()
            calib.fit_ellipsoid(mag, grav)
            calib.accuracy(aligned_data)
            axis = calib.align_along_axis(aligned_data, axis="Y")
            non_linear = calib.apply_non_linear_correction_quick(
                aligned_data, param_count=5
            )
            total_non_linear += non_linear
            total_axis += axis
            print(non_linear, axis)
            # self.assertLess(
            #    non_linear, axis
            # )  # check non_linear process improves accuracy!
        print(total_non_linear / total_axis)

    # we skip this as it is more of a debugging tool than true test
    @staticmethod
    def test_angles_to_matrix_singles():
        angles = [35, +20, 10]

        matrix = Calibration.angles_to_matrix(*angles)
        new_angles = Calibration.matrix_to_angles(matrix)
        np.testing.assert_array_almost_equal(angles, new_angles)

    @staticmethod
    def test_angles_to_matrix_multiple():
        azimuths = np.random.uniform(0, 360, 20)
        inclinations = np.random.uniform(-90, 90, 20)
        rolls = np.random.uniform(-180, 180, 20)

        matrices = Calibration.angles_to_matrix(azimuths, inclinations, rolls)
        new_az, new_inc, new_roll = Calibration.matrix_to_angles(matrices)

        np.testing.assert_array_almost_equal(azimuths, new_az)
        np.testing.assert_array_almost_equal(inclinations, new_inc)
        np.testing.assert_array_almost_equal(rolls, new_roll)

    @unittest.skip
    def test_display_non_linear_maps(self):
        # pylint: disable=invalid-name,import-outside-toplevel,cell-var-from-loop
        import matplotlib.pyplot as plt

        mag, grav, aligned_data = list(self.fixtures.values())[3]
        calib = Calibration()
        calib.fit_ellipsoid(mag, grav)
        calib.accuracy(aligned_data)
        calib.align_along_axis(aligned_data, axis="Y")
        xs = np.linspace(-0.1, 0.1, 30)
        X, Y = np.meshgrid(xs, xs)
        _, axs = plt.subplots(5, 5)
        for i, j in itertools.combinations(range(6), 2):

            def func(x, y):
                params = np.zeros(6)
                params[i] = x
                params[j] = y
                params = np.insert(params, 3, [0, 0, 0])
                calib.mag.set_non_linear_params(params)
                return calib.accuracy(aligned_data) + sum(calib.uniformity(mag, grav))

            f = np.vectorize(func)
            Z = f(X, Y)
            axs[i, j - 1].pcolormesh(X, Y, Z)
        plt.show()

    @unittest.skip
    def test_display_non_linear_terms(self):
        # pylint: disable=invalid-name,import-outside-toplevel,too-many-locals
        import matplotlib.pyplot as plt

        _, axs = plt.subplots(3, len(self.fixtures) // 3, sharex=True, sharey=True)
        params = 5
        for i, (mag, grav, aligned_data) in enumerate(self.fixtures.values()):
            calib = Calibration()
            calib.fit_ellipsoid(mag, grav)
            linear = calib.align_along_axis(aligned_data, axis="Y")
            xs = np.linspace(-1, 1, 40)
            _, min_acc = calib.apply_non_linear_correction(
                aligned_data, param_count=params
            )
            y1 = calib.mag.rbfs[0](xs)
            y2 = calib.mag.rbfs[2](xs)
            q1_acc = calib.apply_non_linear_correction_quick(
                aligned_data, param_count=params
            )
            yq1 = calib.mag.rbfs[0](xs)
            yq2 = calib.mag.rbfs[2](xs)
            q2_acc = calib.apply_non_linear_correction_quick_direct(
                aligned_data, param_count=params
            )
            yq21 = calib.mag.rbfs[0](xs)
            yq22 = calib.mag.rbfs[2](xs)
            ax = axs[i // 3, i % 3]
            ax.plot(xs, y1, "r-", label=f"minimizer({min_acc:.4f})")
            ax.plot(xs, y2, "b-", label=f"(linear({linear:.4f})")
            ax.plot(xs, yq1, "r:", label=f"quick_1({q1_acc:.4f})")
            ax.plot(xs, yq2, "b:")
            ax.plot(xs, yq21, "r--", label=f"quick_2({q2_acc:.4f})")
            ax.plot(xs, yq22, "b--")
            ax.legend()
        plt.show()
