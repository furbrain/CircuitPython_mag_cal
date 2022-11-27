# SPDX-FileCopyrightText: Copyright (c) 2022 Phil Underwood for Underwood Underground
#
# SPDX-License-Identifier: MIT
"""
Various utility functions
"""

try:
    # work with normal numpy
    import numpy as np

    ULAB_PRESENT = False
except ImportError:
    # or work with ulab
    from ulab import numpy as np
    import lstsq

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
    return vector / np.linalg.norm(vector, axis=-1)


class NotCalibrated(Exception):
    """
    Error to raise when calibration has not been done yet
    """
