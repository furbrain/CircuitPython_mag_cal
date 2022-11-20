# SPDX-FileCopyrightText: Copyright (c) 2022 Phil Underwood for Underwood Underground
#
# SPDX-License-Identifier: Unlicense
"""
Implementation of a Radial Basis Function using a gaussian composite.
"""

import numpy as np


class RBF:
    """
    Radial basis function implementation
    """

    def __init__(self, params):
        self.params = np.array(params).reshape((-1, 1))
        if len(params) == 1:
            self.offsets = [0.0]
            self.epsilon = 0.5
        else:
            self.offsets = np.linspace(-1, 1, len(params)).reshape((-1, 1))
            self.epsilon = 1.5 / len(params)

    def __call__(self, x, gaussians=None):
        if gaussians is None:
            gaussians = self.get_gaussians(x)
        result = gaussians.T @ self.params
        return result[:, 0]

    def __str__(self):
        return str(self.params)

    def __repr__(self):
        return str(self)

    def get_gaussians(self, x):
        """
        Get the gaussian offsets for each element in x
        :param x:
        :return:
        """
        x = x.reshape((1, -1))
        distances = x - self.offsets
        gaussians = np.exp(-((distances / self.epsilon) ** 2))
        return gaussians
