import sys

import numpy as np
import pytest

sys.path.append("..")

import src.tools.basis_function.monomials as monomials
import src.tools.diffusion.black_scholes as bs
import src.tools.option.option_main_class as option_main_class
from src.main.dynamic_programming.longstaff_schwarz import multi_dimensional_LS_max
from tools import closed_formula_european_option

tol = 0.5

spot = 42
strike = 50
vol = 0.2
rate = 0.1
T = 1

basis_function = monomials.Monomials(3)


@pytest.mark.parametrize(
    "tol, spot, strike, vol, rate, T, basis_function",
    [
        (0.5, 55, 50, 0.2, 0.1, 1, basis_function),
    ],
)
def test_longstaff_one_dim(
    tol: float,
    spot: float,
    strike: float,
    vol: float,
    rate: float,
    T: float,
    basis_function,
):
    asset = [bs.BlackScholes(spot, rate, vol)]
    call_payoff = lambda x: np.maximum(x - strike, 0)
    option_type = option_main_class.Option(T, strike, rate, call_payoff)
    price_longstaff = multi_dimensional_LS_max(
        asset, option_type, basis_function, 50, int(1e5)
    )[0]
    price_call = closed_formula_european_option(spot, strike, rate, vol, T)
    assert np.abs(price_longstaff - price_call) < tol


@pytest.mark.parametrize(
    "spot, strike, vol, rate, T,dividend, basis_function,d",
    [(110, 100, 0.2, 0.05, 3, 0.10, basis_function, 2)],
)
def test_longstaff_multi_dim(
    spot: float,
    strike: float,
    vol: float,
    rate: float,
    T: float,
    dividend: float,
    basis_function,
    d: float,
):
    asset = [bs.BlackScholes(spot, rate, vol, dividend) for _ in range(d)]
    call_payoff = lambda x: np.maximum(
        np.maximum(x[:, 0] - strike, 0), np.maximum(x[:, 1] - strike, 0)
    )
    option_type = option_main_class.Option(T, strike, rate, call_payoff)
    price_longstaff = multi_dimensional_LS_max(
        asset, option_type, basis_function, 25, int(1e5)
    )[0]
    assert np.abs(price_longstaff - 21.349) < 0.1
