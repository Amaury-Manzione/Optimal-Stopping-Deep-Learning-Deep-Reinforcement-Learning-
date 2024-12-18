from typing import List

import numpy as np
from numpy.core.multiarray import array as array

import src.tools.basis_function.basis_function as basis_function


class Monomials(basis_function.BasisFunction):
    """
    Basis Functions are monomials namemly
    P(X) = 1 + X + X^2  etc
    """

    def __init__(self, degree: int):
        super().__init__(degree)

    def get_matrix(self, X: List[np.array]):
        regr_mat = []
        d = len(X)

        for i in range(1, self.degree):
            for j in range(d):
                regr_mat.append(X[j] ** i)

        for i in range(d):
            for j in range(i + 1, d):
                regr_mat.append(X[i] * X[j])

        regr_mat.append(X[0] ** 0)

        regr_mat = np.array(regr_mat).T

        return regr_mat
