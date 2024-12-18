import sys

sys.path.append("..")

import numpy as np
from scipy.stats import norm


def closed_formula_european_option(spot, strike, rate, vol, maturity):
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * maturity) / (
        vol * np.sqrt(maturity)
    )
    d2 = d1 - vol * np.sqrt(maturity)

    return spot * norm.cdf(d1) - strike * np.exp(-rate * maturity) * norm.cdf(d2)
