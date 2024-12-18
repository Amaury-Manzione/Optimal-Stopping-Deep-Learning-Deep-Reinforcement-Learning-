import sys
import time
from typing import List, Union

import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm

import src.main.dynamic_programming.dp_pricer as dp_pricer
import src.tools.basis_function.basis_function as basis_function
import src.tools.diffusion.ito_process as ito_process
import src.tools.option.option_main_class as option_main_class


class PricerLeastSquaresLongstaffSchwarz(dp_pricer.PricerDynamicProgramming):
    def __init__(
        self,
        option_: option_main_class.Option,
        asset_: List[ito_process.ItoProcess],
        fun: basis_function.BasisFunction,
    ):
        super().__init__(option_, asset_)
        self.basis_function = fun
        self.models = {"underlying": [], "cash_flows": [], "coefficients": []}

    def get_strategy(
        self,
        data: np.ndarray,
        current_discount_vect: np.ndarray,
        cash_flows: np.ndarray,
        **kwargs
    ):
        _, d = data.shape
        X = [data[:, i] for i in range(d)]
        regr_mat = self.basis_function.get_matrix(X)

        intrinsic_value = np.squeeze(self.option_type.payoff(data))
        discount_vect = current_discount_vect
        ydata = cash_flows * discount_vect

        a = linalg.lstsq(np.squeeze(regr_mat), ydata)[0]
        continuation_value = np.dot(
            np.squeeze(regr_mat), a
        )  # approximation for the conditional expectation

        self.models["underlying"].append(np.squeeze(data))
        self.models["cash_flows"].append(ydata)
        self.models["coefficients"].append(a)

        mse = np.mean((ydata - continuation_value) ** 2)
        index_Exercise = np.argwhere(
            np.squeeze(intrinsic_value) > np.squeeze(continuation_value)
        )

        return index_Exercise, np.array(mse)
