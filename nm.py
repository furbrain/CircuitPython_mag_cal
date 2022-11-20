# SPDX-FileCopyrightText: Copyright (c) 2022 Phil Underwood for Underwood Underground
#
# SPDX-License-Identifier: Unlicense
"""
This is an implementation of the nelder-mead optimisation algorithm, adapted from
`scipy.optimize.fmin`.
"""

try:
    import numpy as np
except ImportError:
    from ulab import numpy as np


RHO = 1
CHI = 2
PSI = 0.5
SIGMA = 0.5

NONZDELT = 0.05
ZDELT = 0.00025


class Minimizer:
    """
    Minimization of scalar function of one or more variables using the
    Nelder-Mead algorithm.
    Options
    -------
    maxiter: int
        Maximum allowed number of iterations.
        Will default to ``dims*200``, where ``dims`` is the number of
        variables.
    xatol : float, optional
        Absolute error in xopt between iterations that is acceptable for
        convergence.
    fatol : number, optional
        Absolute error in func(xopt) between iterations that is acceptable for
        convergence.
    References
    ----------
    .. [1] Gao, F. and Han, L.
       Implementing the Nelder-Mead simplex algorithm with adaptive
       parameters. 2012. Computational Optimization and Applications.
       51:1, pp. 259-277
    """

    # pylint: disable=invalid-name
    def __init__(
        self,
        func,
        x0,
        xatol=1e-4,
        fatol=1e-4,
    ):

        # create the initial simplex
        self.dims = len(x0)
        self.sim = np.empty((self.dims + 1, self.dims))
        self.sim[0] = x0
        self.xatol = xatol
        self.fatol = fatol
        self.func = func
        for k in range(self.dims):
            y = x0[:]
            if y[k] != 0:
                y[k] = (1 + NONZDELT) * y[k]
            else:
                y[k] = ZDELT
            self.sim[k + 1] = y

        # If neither are set, then set both to default
        self.fsim = np.full((self.dims + 1,), np.inf)
        for k in range(self.dims + 1):
            self.fsim[k] = func(self.sim[k])

    def optimize(self, maxiter=None):
        """
        Perform the optimization.
        :param maxiter: How many cycles to run before giving up
        :return: Dictionary containing the following elements:
          status
            "Success" if solution has converged or "Max Iterations" if not
          x
            The optimised solution
          simplex
            The simplex at time of termination
          fsim
            The function evaluation for each point on the simplex
          iterations
            Number of iterations completed
        """
        if maxiter is None:
            maxiter = self.dims * 200
        iterations = 1
        while iterations < maxiter:
            order = np.argsort(self.fsim, axis=0)
            best = order[0]
            worst = order[-1]
            second_worst = order[-2]
            sim_worst = self.sim[worst]
            if (np.max(abs(self.sim - self.sim[best]))) <= self.xatol and np.max(
                abs(self.fsim - self.fsim[best])
            ) <= self.fatol:
                break

            # calculate centroid, by calculating sum of all vertices, minus the worst
            xbar = (np.sum(self.sim, axis=0) - sim_worst) / self.dims
            xr = (1 + RHO) * xbar - RHO * sim_worst
            fxr = self.func(xr)

            if fxr < self.fsim[best]:
                self._extend(fxr, worst, xbar, xr)
            else:  # fsim[best] <= fxr
                if fxr < self.fsim[second_worst]:
                    self.sim[worst] = xr
                    self.fsim[worst] = fxr
                else:  # fxr >= fsim[-2]
                    # Perform contraction
                    doshrink = self._contract(fxr, sim_worst, worst, xbar)
                    if doshrink:
                        self._shrink(best)
            iterations += 1

        x = self.sim[best]

        if iterations >= maxiter:
            msg = "Max Iterations"
        else:
            msg = "Success"

        return {
            "status": msg,
            "x": x,
            "simplex": self.sim,
            "fsim": self.fsim,
            "iterations": iterations,
        }

    def _shrink(self, best):
        for j in range(self.dims + 1):
            if j != best:
                self.sim[j] = self.sim[best] + SIGMA * (self.sim[j] - self.sim[best])
                self.fsim[j] = self.func(self.sim[j])

    def _contract(self, fxr, sim_worst, worst, xbar):
        if fxr < self.fsim[worst]:
            xc = (1 + PSI * RHO) * xbar - PSI * RHO * self.sim[worst]
            fxc = self.func(xc)
            if fxc <= fxr:
                self.sim[worst] = xc
                self.fsim[worst] = fxc
                return False
        else:
            # Perform an inside contraction
            xcc = (1 - PSI) * xbar + PSI * sim_worst
            fxcc = self.func(xcc)
            if fxcc < self.fsim[worst]:
                self.sim[worst] = xcc
                self.fsim[worst] = fxcc
                return False
        return True

    def _extend(self, fxr, worst, xbar, xr):
        xe = (1 + RHO * CHI) * xbar - RHO * CHI * self.sim[worst]
        fxe = self.func(xe)
        if fxe < fxr:
            self.sim[worst] = xe
            self.fsim[worst] = fxe
        else:
            self.sim[worst] = xr
            self.fsim[worst] = fxr


class CheckOptimize:
    """Base test case for a simple constrained entropy maximization problem
    (the machine translation example of Berger et al in
    Computational Linguistics, vol 22, num 1, pp 39--72, 1996.)
    """

    # pylint: disable=invalid-name
    def __init__(self):
        self.F = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [1, 0, 0]])
        self.K = np.array([1.0, 0.3, 0.5])
        self.startparams = np.zeros(3)
        self.solution = np.array([0.0, -0.524869316, 0.487525860])
        self.maxiter = 1000
        self.funccalls = 0
        self.gradcalls = 0
        self.trace = []

    def func(self, x):
        """
        The actual function
        :param x:
        :return:
        """
        self.funccalls += 1
        if self.funccalls > 6000:
            raise RuntimeError("too many iterations in optimization routine")
        log_pdot = np.dot(self.F, x)
        logZ = np.log(sum(np.exp(log_pdot)))
        f = logZ - np.dot(self.K, x)
        self.trace.append(x[:])
        return f


chk = CheckOptimize()
optimizer = Minimizer(chk.func, np.array((1.0, 1.0, 1.0)))
res = optimizer.optimize()
print(res)
print(chk.funccalls)
