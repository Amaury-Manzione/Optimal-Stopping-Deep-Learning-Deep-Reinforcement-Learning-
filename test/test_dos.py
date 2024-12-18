import sys

import numpy as np
import pytest
import torch
from torch import nn

sys.path.append("..")

import src.main.dynamic_programming.dos as dos
import src.tools.diffusion.black_scholes as bs
import src.tools.option.option_main_class as option_main_class
from tools import closed_formula_european_option

tol = 1e-2

spot = 42
strike = 50
vol = 0.2
rate = 0.1
T = 1

learning_rate = 1e-3
epochs = 300
batch = 250
N_training = 1000
N_test = 100000
n_simulation = 10
list_weights = [50, 30]
activation_function = nn.ReLU()


@pytest.mark.parametrize(
    "tol, spot, strike, vol, rate, T",
    [
        (0.5, 65, 50, 0.2, 0.1, 1),
    ],
)
def test_dos(
    tol: float,
    spot: float,
    strike: float,
    vol: float,
    rate: float,
    T: float,
):
    asset = [bs.BlackScholes(spot, rate, vol)]
    call_payoff = lambda x: torch.maximum(torch.squeeze(x) - strike, torch.tensor(0))
    option_type = option_main_class.Option(T, strike, rate, call_payoff)
    dos_object = dos.DOS(asset, option_type, n_simulation)
    mlps = dos_object.train_dos(
        N_training,
        learning_rate,
        epochs,
        batch,
        activation_function,
        list_weights,
        verbose=False,
    )[1]
    price_dos = dos_object.test_dos(mlps, N_test)[0]
    price_call = closed_formula_european_option(spot, strike, rate, vol, T)
    assert np.abs(price_dos - price_call) < tol
