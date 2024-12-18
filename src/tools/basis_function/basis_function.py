from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np


class BasisFunction(metaclass=ABCMeta):
    """
    Basis function for approximating the
    conditional expectation
    """

    def __init__(self, degree: int):
        self.degree = degree

    @abstractmethod
    def get_matrix(self, X: List[np.array]):
        """
        return matrix of 1, P(X), P(X^2) ,P(Y), P(XY) etc for
        linear Regression

        X : list of underlyings in dimension d>=2
        (X,Y,Z,etc)
        """
        raise Exception(
            "get_matrix() method is not implemented in one of the subclasses of Pricer"
        )

    def get_polynom(self, x, coefficients: np.ndarray):
        x = list(x)
        monoms = self.get_matrix(x)
        return np.dot(np.squeeze(monoms), coefficients)
