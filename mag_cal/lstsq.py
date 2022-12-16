# SPDX-FileCopyrightText: Copyright (c) 2022 Phil Underwood for Underwood Underground
#
# SPDX-License-Identifier: Unlicense
"""
This provides a least squares implementation for use with `micropython-ulab`. This has been
adapted from the `numpy.linalg.lstsq` implementation.
"""

from ulab import numpy as np

# pylint: disable=invalid-name
def lstsq(a, b):
    r"""
    Return the least-squares solution to a linear matrix equation.
    Computes the vector `x` that approximately solves the equation
    ``a @ x = b``. The equation may be under-, well-, or over-determined
    (i.e., the number of linearly independent rows of `a` can be less than,
    equal to, or greater than its number of linearly independent columns).
    If `a` is square and of full rank, then `x` (but for round-off error)
    is the "exact" solution of the equation. Else, `x` minimizes the
    Euclidean 2-norm :math:`||b - ax||`. If there are multiple minimizing
    solutions, the one with the smallest 2-norm :math:`||x||` is returned.
    Parameters
    ----------
    a : (M, dims) array_like
        "Coefficient" matrix.
    b : {(M,), (M, K)} array_like
        Ordinate or "dependent variable" values. If `b` is two-dimensional,
        the least-squares solution is calculated for each of the `K` columns
        of `b`.
    Returns
    -------
    x : {(dims,), (dims, K)} ndarray
        Least-squares solution. If `b` is two-dimensional,
        the solutions are in the `K` columns of `x`.
    residuals : {(1,), (K,), (0,)} ndarray
        Sums of squared residuals: Squared Euclidean 2-norm for each column in
        ``b - a @ x``.
        If the rank of `a` is < dims or M <= dims, this is an empty array.
        If `b` is 1-dimensional, this is a (1,) shape array.
        Otherwise the shape is (K,).
    Raises
    ------
    LinAlgError
        If computation does not converge.
    See Also
    --------
    scipy.linalg.lstsq : Similar function in SciPy.
    Notes
    -----
    If `b` is a matrix, then all array results are returned as matrices.
    Examples
    --------
    Fit a line, ``y = mx + c``, through some noisy data-points:
    >>> x = np.array([0, 1, 2, 3])
    >>> y = np.array([-1, 0.2, 0.9, 2.1])
    By examining the coefficients, we see that the line should have a
    gradient of roughly 1 and cut the y-axis at, more or less, -1.
    We can rewrite the line equation as ``y = Ap``, where ``A = [[x 1]]``
    and ``p = [[m], [c]]``.  Now use `lstsq` to solve for `p`:
    >>> A = np.vstack([x, np.ones(len(x))]).T
    >>> A
    array([[ 0.,  1.],
           [ 1.,  1.],
           [ 2.,  1.],
           [ 3.,  1.]])
    >>> m, c = np.linalg.lstsq(A, y)[0]
    >>> m, c
    (1.0 -0.95) # may vary
    """

    # check shapes
    is_1d = len(b.shape) == 1
    if is_1d:
        b = b.reshape((b.shape[0], 1))
    if len(a.shape) != 2:
        raise ValueError("Matrix a must be two dimensional")
    if len(b.shape) != 2:
        raise ValueError("Matrix b must be one or two dimensional")
    m, n = a.shape[-2:]
    m2, _ = b.shape[-2:]
    if m != m2:
        raise ValueError(f"Incompatible dimensions: {a.shape}, {b.shape}")

    if m > n:
        # system is overdetermined or at least appears to be so
        q, r = np.linalg.qr(a)
        x = np.dot(np.dot(np.linalg.inv(r), q.transpose()), b)
    else:
        # system is underdetermined
        q, r = np.linalg.qr(a.transpose())
        x = np.dot(q, np.dot(np.linalg.inv(r.transpose()), b))

    resids = np.linalg.norm(np.dot(a, x) - b, axis=0)
    resids *= resids

    # remove the axis we added
    if is_1d:
        x = x.flatten()
        resids = resids.flatten()
    return x, resids
