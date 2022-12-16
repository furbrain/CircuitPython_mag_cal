# SPDX-FileCopyrightText: Copyright (c) 2022 Phil Underwood for Underwood Underground
# SPDX-FileCopyrightText: Copyright (c) 2000 Travis E Oliphatn
#
# SPDX-License-Identifier: Unlicense
"""
This is an implementation of the nelder-mead optimisation algorithm, adapted from
`scipy.optimize.fmin`, written by Travis E Oliphant
"""

try:
    import numpy as np
except ImportError:
    from ulab import numpy as np

NONZDELT = 0.05
ZDELT = 0.00025


class Minimizer:
    """
    Minimization of scalar function of one or more variables using the
    Nelder-Mead algorithm.

    References
    ----------
    .. [1] Gao, F. and Han, L.
       Implementing the Nelder-Mead simplex algorithm with adaptive
       parameters. 2012. Computational Optimization and Applications.
       51:1, pp. 259-277
    """

    # pylint: disable=invalid-name,too-many-arguments
    def __init__(
        self,
        func,
        x0,
        adaptive=False,
        xatol=1e-4,
        fatol=1e-4,
    ):
        """
        Create a Minimizer object, you can then call with optimize to get results.

        :param callable func: Function to be called, must return a single float
        :param np.ndarray x0: Initial value to start from
        :param bool adaptive: Whether to use adaptive methods - can be useful when solving
          higher dimensional problems
        :param float xatol: Absolute error in xopt between iterations that is acceptable for
          convergence. Default is 0.0001
        :param float fatol: Absolute error in func(xopt) between iterations that is acceptable for
        convergence. Default is 0.0001
        """
        # create the initial simplex
        self.dims = len(x0)
        if adaptive:
            dim = float(len(x0))
            self.rho = 1
            self.chi = 1 + 2 / dim
            self.psi = 0.75 - 1 / (2 * dim)
            self.sigma = 1 - 1 / dim
        else:
            self.rho = 1
            self.chi = 2
            self.psi = 0.5
            self.sigma = 0.5

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
        :param maxiter: How many cycles to run before giving up, will default to dims*200 if not
          specified
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
            xr = (1 + self.rho) * xbar - self.rho * sim_worst
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
            msg = "max iterations"
        else:
            msg = "success"

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
                self.sim[j] = self.sim[best] + self.sigma * (
                    self.sim[j] - self.sim[best]
                )
                self.fsim[j] = self.func(self.sim[j])

    def _contract(self, fxr, sim_worst, worst, xbar):
        if fxr < self.fsim[worst]:
            xc = (1 + self.psi * self.rho) * xbar - self.psi * self.rho * self.sim[
                worst
            ]
            fxc = self.func(xc)
            if fxc <= fxr:
                self.sim[worst] = xc
                self.fsim[worst] = fxc
                return False
        else:
            # Perform an inside contraction
            xcc = (1 - self.psi) * xbar + self.psi * sim_worst
            fxcc = self.func(xcc)
            if fxcc < self.fsim[worst]:
                self.sim[worst] = xcc
                self.fsim[worst] = fxcc
                return False
        return True

    def _extend(self, fxr, worst, xbar, xr):
        xe = (1 + self.rho * self.chi) * xbar - self.rho * self.chi * self.sim[worst]
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


if __name__ == "__main__":
    chk = CheckOptimize()
    optimizer = Minimizer(chk.func, np.array((1.0, 1.0, 1.0)))
    res = optimizer.optimize()
    print(res)
    print(chk.funccalls)
