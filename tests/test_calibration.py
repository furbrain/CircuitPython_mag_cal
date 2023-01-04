# SPDX-FileCopyrightText: Copyright (c) 2022 Phil Underwood for Underwood Underground
#
# SPDX-License-Identifier: MIT
import itertools
import unittest
from pathlib import Path
from unittest import TestCase

import numpy as np

from mag_cal.calibration import Calibration
from mag_cal.utils import read_fixture


class TestCalibration(TestCase):
    def setUp(self) -> None:
        np.set_printoptions(precision=4, suppress=True)
        self.fixtures = {}
        for fname in (Path(__file__).parent / "fixtures" / "cal_data").glob("*.json"):
            with open(fname) as fixture:
                aligned_data, grav, mag = read_fixture(fixture.read())
                self.fixtures[fname] = (mag, grav, aligned_data)

    def test_fit_ellipsoid(self):
        for mag, grav, _ in self.fixtures.values():
            calib = Calibration()
            mag_accuracy, grav_accuracy = calib.fit_ellipsoid(mag, grav)
            self.assertGreater(0.01, mag_accuracy)
            self.assertGreater(0.001, grav_accuracy)

    def test_fit_to_axis(self):
        for mag, grav, aligned_data in self.fixtures.values():
            calib = Calibration()
            calib.fit_ellipsoid(mag, grav)
            before = calib.accuracy(aligned_data)
            calib.fit_to_axis(aligned_data, axis="Y")
            after = calib.accuracy(aligned_data)
            print(before, after)
            # we should expect accuracy to at least double after alignment
            self.assertLess(2.0, before / after)

    def test_non_linear(self):
        non_linears = []
        ratios = []
        for mag, grav, aligned_data in self.fixtures.values():
            calib = Calibration()
            calib.fit_ellipsoid(mag, grav)
            calib.accuracy(aligned_data)
            before = calib.fit_to_axis(aligned_data, axis="Y")
            _, after = calib.fit_non_linear(
                aligned_data, sensor=Calibration.MAGNETOMETER, param_count=3
            )
            non_linears.append(after)
            ratios.append(after / before)
            print(after, before)
            print(after, before)
            self.assertLess(
                after, before
            )  # check non_linear process improves accuracy!
        print("Avg Accuracy:", np.mean(np.array(non_linears)))
        print("Improvement: ", np.mean(np.array(ratios)))

    def test_non_linear_quick(self):
        non_linears = []
        ratios = []
        for mag, grav, aligned_data in self.fixtures.values():
            calib = Calibration()
            calib.fit_ellipsoid(mag, grav)
            calib.accuracy(aligned_data)
            before = calib.fit_to_axis(aligned_data, axis="Y")
            after = calib.fit_non_linear_quick(aligned_data, param_count=5)
            non_linears.append(after)
            ratios.append(after / before)
            print(before, after)
        print("Avg Accuracy:", np.mean(np.array(non_linears)))
        improvement = np.mean(np.array(ratios))
        print("Improvement: ", improvement)
        self.assertLess(improvement, 0.75)

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
        calib.fit_to_axis(aligned_data, axis="Y")
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
            linear = calib.fit_to_axis(aligned_data, axis="Y")
            xs = np.linspace(-1, 1, 40)
            _, min_acc = calib.fit_non_linear(aligned_data, param_count=params)
            y1 = calib.mag.rbfs[0](xs)
            y2 = calib.mag.rbfs[2](xs)
            q1_acc = calib.fit_non_linear_quick(aligned_data, param_count=params)
            yq1 = calib.mag.rbfs[0](xs)
            yq2 = calib.mag.rbfs[2](xs)
            yq3 = calib.mag.rbfs[1](xs)
            ax = axs[i // 3, i % 3]
            ax.plot(xs, y1, "r-", label=f"minimizer({min_acc:.4f})")
            ax.plot(xs, y2, "b-", label=f"(linear({linear:.4f})")
            ax.plot(xs, yq1, "r:", label=f"quick_1({q1_acc:.4f})")
            ax.plot(xs, yq2, "b:")
            ax.plot(xs, yq3, "g:")
            ax.legend()
        plt.show()
