import sys
import time
from functools import wraps
from typing import List, Union

import numpy as np
import torch
from tqdm import tqdm

import src.main.pricer as pricer
import src.tools.diffusion.ito_process as ito_process
import src.tools.option.option_main_class as option_main_class


class PricerDynamicProgramming(pricer.Pricer):
    def __init__(
        self, option_: option_main_class.Option, asset_: List[ito_process.ItoProcess]
    ):
        super().__init__(option_, asset_)
        self.intermediate_values = {}

    def get_strategy(
        self,
        data: np.ndarray,
        exercice_time: np.ndarray,
        current_discount_vect: np.ndarray,
        cash_flows: np.ndarray,
        intrinsic_value: np.ndarray,
        **kwargs
    ):
        raise Exception(
            "price() method is not implemented in one of the subclasses of Pricer"
        )

    def price(self, n: int, N: int, seed=None, disable=True, **kwargs):
        start_time = time.time()
        self.intermediate_values = {"step": [], "loss": []}
        maturity = self.option_type.T
        r = self.option_type.r

        dt = float(maturity / (n - 1))
        d = len(self.asset)

        paths = np.zeros((N, n, d))
        for i in range(d):
            asset_i = self.asset[i]
            paths[:, :, i] = asset_i.get_path(n, dt, N, seed)

        cash_flows = self.option_type.payoff(paths[:, n - 1, :])

        cash_flows = np.squeeze(cash_flows)

        discountVect = np.exp(-r * dt * np.arange(1, n, 1))

        exercice_time = (n - 1) * np.ones(N)
        exercice_time = exercice_time.astype("int")

        loop = tqdm(range(n - 2, 0, -1), disable=disable)
        for step in loop:
            intrinsic_value = np.squeeze(self.option_type.payoff(paths[:, step, :]))

            current_discount_vect = discountVect[exercice_time - step - 1]

            index_Exercise, error = self.get_strategy(
                data=paths[:, step, :],
                # exercice_time=current_discount_vect,
                current_discount_vect=current_discount_vect,
                cash_flows=cash_flows,
                intrinsic_value=intrinsic_value,
            )
            cash_flows[index_Exercise] = intrinsic_value[index_Exercise]
            exercice_time[index_Exercise] = step
            self.intermediate_values["step"].append(step)
            self.intermediate_values["loss"].append(error)
            # loop.set_postfix(loss=error[-1])

        payoff = cash_flows * discountVect[exercice_time - 1]
        price = np.mean(payoff)

        std = np.std(payoff)  # standard deviation estimator

        error = 1.96 * std / np.sqrt(N)

        ci_up = price + error
        ci_down = price - error

        t = time.time() - start_time

        return price, ci_up, ci_down, t, exercice_time
