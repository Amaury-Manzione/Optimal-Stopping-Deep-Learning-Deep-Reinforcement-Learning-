import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append("..\..\..")

import src.main.nos.nos as nos_pricer
import src.tools.diffusion.black_scholes as bs
import src.tools.option.option_main_class as option_main_class

spot = 100
strike = 100
vol = 0.2
rate = 0.05
dividend = 0.10
T = 3

weights = [40, 30]
batch_size = 5000
learning_rate = 1e-2
epochs = 150
activation_function = torch.nn.ReLU()


n = 50
N = 100000


epsilon = 7
l_girsanov = rate / vol


def change_coordinates(x):
    return torch.max(x, dim=1).values


def call_payoff(x):
    max_mat = torch.maximum(x - strike, torch.tensor(0.0))  # Broadcasting over strike

    return torch.max(max_mat, dim=1).values


put_payoff = lambda x: torch.maximum(torch.tensor(strike) - x, torch.tensor(0))


asset1 = [bs.BlackScholes(spot, rate, vol) for i in range(1)]
asset2 = [bs.BlackScholes(spot, rate, vol, dividend) for i in range(50)]
option1 = option_main_class.Option(T, strike, rate, put_payoff)
option2 = option_main_class.Option(T, strike, rate, call_payoff)

pricer_nos = nos_pricer.NOS(
    option2,
    asset2,
    change_coordinates=change_coordinates,
    epsilon=epsilon,
    l_girsanov=l_girsanov,
    weights=weights,
    last_bias=strike,
    activation_function=activation_function,
    learning_rate=learning_rate,
    epochs=epochs,
    batch_size=batch_size,
)

history, _, _ = pricer_nos.find_optimal_region(n, N, disable=False)

plt.plot(history.numpy(), label=f"learning rate : {learning_rate}")
plt.ylabel("loss function")
plt.xlabel("epochs")
plt.legend()
plt.show()
