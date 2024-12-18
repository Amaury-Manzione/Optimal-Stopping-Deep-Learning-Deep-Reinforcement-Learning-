from typing import List

import numpy as np
from numpy.core.multiarray import array as array

import src.tools.basis_function.basis_function as basis_function


class Laguerre(basis_function.BasisFunction):
    """
    Basis Functions are laguerre polynomials
    """

    def __init__(self, degree: int):
        super().__init__(degree)

    def get_matrix(self, X: List[np.array]):
        RegrMat = []
        d = len(X)

        for j in range(d):
            RegrMat.append(X[j] ** 0)
            for i in range(1, self.degree):
                if i == 1:
                    RegrMat.append(X[j])
                else:
                    val = (
                        -(2 * i - 1 - X[j]) * RegrMat[i - 1] - (i - 1) * RegrMat[i - 2]
                    ) / i
                    RegrMat.append(val)

        for i in range(d):
            for j in range(i + 1, d):
                RegrMat.append(X[i] * X[j])

        RegrMat = np.array(RegrMat).T
        return RegrMat
