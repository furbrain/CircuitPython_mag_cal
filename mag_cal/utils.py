# SPDX-FileCopyrightText: Copyright (c) 2022 Phil Underwood for Underwood Underground
#
# SPDX-License-Identifier: MIT
"""
Various utility functions
"""

import json
import math

try:
    # work with normal numpy
    import numpy as np

    ULAB_PRESENT = False

except ImportError:
    # or work with ulab
    from ulab import numpy as np
    from . import lstsq

    ULAB_PRESENT = True


def solve_least_squares(A: np.ndarray, B: np.ndarray):
    # pylint: disable=invalid-name
    """
    Calculate x such that Ax=B. Convenience function so can use either `numpy` or `ulab`

    :param A:
    :param B:
    :return: x
    """
    if ULAB_PRESENT:
        # pylint: disable=used-before-assignment
        coeffs, _ = lstsq.lstsq(A, B)
    else:
        coeffs, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    return coeffs


def normalise(vector: np.ndarray):
    """
    Convert a vector or array of vectors to unit length
    :param np.ndarray vector:  Input vector(s)
    :return: Normalised vector(s)
    """
    return (vector.transpose() / np.linalg.norm(vector, axis=-1)).transpose()


class NotCalibrated(Exception):
    """
    Error to raise when calibration has not been done yet
    """


def read_fixture(fixture: str):
    """
    Read a fixture file and return magnetic and accelerometer readings
    :param fixture: json text from a fixture file
    :return:
    """
    data = json.loads(fixture)
    mag = data["shots"]["mag"]
    grav = data["shots"]["grav"]
    mag = [x for x in mag if x is not None and not math.isnan(x)]
    mag = np.array(mag)
    mag = mag.reshape((mag.shape[0] // 3, 3))
    grav = [x for x in grav if x is not None and not math.isnan(x)]
    grav = np.array(grav)
    grav = grav.reshape((grav.shape[0] // 3, 3))
    mag_aligned = [mag[8:16], mag[16:24]]
    grav_aligned = [grav[8:16], grav[16:24]]
    aligned_data = list(zip(mag_aligned, grav_aligned))
    return aligned_data, grav, mag


def cross(a: np.ndarray, b: np.ndarray):
    # pylint: disable=invalid-name
    """
    Return a x b where x is the cross-product operator
    :param np.ndarray a: Numpy array of shape (3,) or (N,3)
    :param np.ndarray b: Numpy array of shape (3,) or (N,3)
    :return: a x b
    """
    len_a = len(a.shape)
    len_b = len(b.shape)
    if len_a == 1 and len_b == 1:
        return np.cross(a, b)
    if len_a == 1:
        return np.array([np.cross(a, x) for x in b])
    if len_b == 1:
        return np.array([np.cross(x, b) for x in a])
    return np.array([np.cross(x, y) for x, y in zip(a, b)])
