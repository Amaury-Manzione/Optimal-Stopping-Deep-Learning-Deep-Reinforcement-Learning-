import sys

import torch
from torch import nn

sys.path.append("..\..\..")
import src.main.delta_v_dqn.agent as agent
import src.main.delta_v_dqn.delta_v_dqn as dqn
import src.tools.diffusion.black_scholes as bs
import src.tools.option.option_main_class as option_main_class

strike = 1
r = 0.06
vol = 0.2
spot = 36 / 40
maturity = 1

list_weights = [20]
activation_function = nn.ReLU()
n_simulation = 50
epsilon = 9
epsilon_decay = 0.05
epsilon_min = 0.1
learning_rate = 1e-2
d = 1
batch_size = 1000
replay_memory_init = 15000
replay_memory_capacity = 100000
N_update = 100


myagent = agent.Agent(
    epsilon,
    epsilon_decay,
    epsilon_min,
    replay_memory_init,
    replay_memory_capacity,
    N_update,
    learning_rate,
    batch_size,
    activation_function,
    list_weights,
    d,
)

process = bs.BlackScholes(spot, r, vol)
put_option_payoff = lambda x: max(strike - x, 0)
option_ = option_main_class.Option(maturity, strike, r, put_option_payoff)


dqn_object = dqn.delta_v_dqn(process, option_, myagent)

episodes_pre_train = 2000
episodes_train = 3000

dqn_object.train(n_simulation, episodes_pre_train, episodes_train)
