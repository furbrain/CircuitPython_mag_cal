# SPDX-FileCopyrightText: Copyright (c) 2022 Phil Underwood for Underwood Underground
#
# SPDX-License-Identifier: MIT
from unittest import TestCase

import numpy as np

from mag_cal.axes import Axes


class TestAxes(TestCase):
    def test_get_a_and_p_good_values(self):
        # pylint: disable=protected-access
        with self.subTest("Positive indices"):
            with self.subTest("Upper case"):
                for i, char in enumerate("XYZ"):
                    self.assertTupleEqual(
                        (i, +1), Axes._get_axis_and_polarity(f"+{char}")
                    )
            with self.subTest("Lower case"):
                for i, char in enumerate("xyz"):
                    self.assertTupleEqual(
                        (i, +1), Axes._get_axis_and_polarity(f"+{char}")
                    )
        with self.subTest("Negative indices"):
            with self.subTest("Upper case"):
                for i, char in enumerate("XYZ"):
                    self.assertTupleEqual(
                        (i, -1), Axes._get_axis_and_polarity(f"-{char}")
                    )
            with self.subTest("Lower case"):
                for i, char in enumerate("xyz"):
                    self.assertTupleEqual(
                        (i, -1), Axes._get_axis_and_polarity(f"-{char}")
                    )

    def test_get_a_and_p_short_values(self):
        # pylint: disable=protected-access
        self.assertRaises(ValueError, Axes._get_axis_and_polarity, "+")
        self.assertRaises(ValueError, Axes._get_axis_and_polarity, "X")

    def test_get_a_and_p_long_values(self):
        # pylint: disable=protected-access
        self.assertRaises(ValueError, Axes._get_axis_and_polarity, "+xx")
        self.assertRaises(ValueError, Axes._get_axis_and_polarity, "++x")

    def test_get_a_and_p_wrong_values(self):
        # pylint: disable=protected-access
        self.assertRaises(ValueError, Axes._get_axis_and_polarity, "+w")
        self.assertRaises(ValueError, Axes._get_axis_and_polarity, "[x")

    def test_init_too_short(self):
        self.assertRaises(ValueError, Axes, "+x+Y+")

    def test_init_too_long(self):
        self.assertRaises(ValueError, Axes, "+x+Y+Z+")

    def test_init_bad_values(self):
        self.assertRaises(ValueError, Axes, "+X+X+Y")
        self.assertRaises(ValueError, Axes, "+Y+Y+Z")
        self.assertRaises(ValueError, Axes, "+Z+Z+X")

    def test_init_good_values(self):
        fields = [
            (("+x+Y+Z"), (0, 1, 2), (1, 1, 1)),
            (("-x-Y-Z"), (0, 1, 2), (-1, -1, -1)),
            (("-Z-X+Y"), (2, 0, 1), (-1, -1, 1)),
        ]
        for text, pos, pol in fields:
            with self.subTest(f'Testing "{text}" gives axes {pos} and polarity {pol}'):
                axes = Axes(text)
                self.assertSequenceEqual(axes.indices, pos)
                self.assertSequenceEqual(axes.polarities, pol)

    def test_fix_axes_noop(self):
        # pylint: disable=no-self-use
        """
        Test using multiple numbers of dimensions
        :return:
        """
        axes = Axes("+X+Y+Z")
        for _ in range(5):
            r = np.random.random(3)
            np.testing.assert_equal(r, axes.fix_axes(r))
        for _ in range(5):
            r = np.random.random((5, 3))
            np.testing.assert_equal(r, axes.fix_axes(r))
        for _ in range(5):
            r = np.random.random((4, 5, 3))
            np.testing.assert_equal(r, axes.fix_axes(r))

    def test_fix_axes_jiggle(self):
        # pylint: disable=no-self-use
        axes = Axes("+Y+Z+X")
        tests = [
            ((2, 3, 4), (3, 4, 2)),
            (((2, 3, 4), (1, 5, 7)), ((3, 4, 2), (5, 7, 1))),
        ]
        for orig, fixed in tests:
            np.testing.assert_equal(fixed, axes.fix_axes(orig))

    def test_fix_axes_jiggle_invert(self):
        # pylint: disable=no-self-use
        axes = Axes("+Y-Z-X")
        tests = [
            ((2, 3, 4), (3, -4, -2)),
            (((2, 3, 4), (1, 5, 7)), ((3, -4, -2), (5, -7, -1))),
        ]
        for orig, fixed in tests:
            np.testing.assert_equal(fixed, axes.fix_axes(orig))

    def test___str__(self):
        tests = ["+X+Y+Z", "+x+y+z", "+y+x+z", "-x+y-z", "-z+x-Y"]
        for test in tests:
            self.assertEqual(test.upper(), str(Axes(test)))
