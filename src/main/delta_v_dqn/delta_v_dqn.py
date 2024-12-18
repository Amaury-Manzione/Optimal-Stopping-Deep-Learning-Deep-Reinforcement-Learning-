import sys
import time as time
from typing import Callable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.append("..\..")

import src.main.delta_v_dqn.agent as agent
import src.tools.diffusion.ito_process as ito_process
import src.tools.option.option_main_class as option_main_class


class delta_v_dqn:
    """Class representing a bermudan option priced with deep q-learning paradigm."""

    def __init__(
        self,
        process: ito_process.ItoProcess,
        option_type: option_main_class.Option,
        myagent: agent.Agent,
    ):
        """Construct an object of type bermudan.

        Parameters
        ----------
        process : ito_process.ItoProcess
            underlying process
        option_type : option.Option
            vanilla option (type specified by its payoff)
        myagent : agent.Agent
            dqn agent
        """
        self.process = process
        self.option_type = option_type
        self.myagent = myagent

    def train(
        self,
        n_simulation: int,
        n_episodes: int,
        start_training: int,
        disable=True,
        seed=None,
    ):
        # timestep
        dt = float(self.option_type.T / (n_simulation - 1))

        # loading the environment
        paths = self.process.get_path(n_simulation, dt, n_episodes, seed)
        paths = torch.tensor(paths, dtype=torch.double)

        print("training")
        # history of rewards per episode
        history_reward_train = []
        history_losses_train = np.zeros(n_episodes)
        history_pos_train = torch.zeros(n_episodes)
        history_sigma = np.zeros(n_episodes)

        count_target_update = 0

        for episode in tqdm(
            range(n_episodes),
            disable=disable,
        ):
            continue_ = True
            current_time = 1
            while continue_:
                input_ = torch.tensor(
                    [current_time, paths[episode, current_time]],
                    dtype=torch.double,
                )
                self.myagent.continuing_network.eval()
                Q_continuing = self.myagent.continuing_network(input_.unsqueeze(0))
                r_stopping = torch.exp(
                    torch.tensor(-self.option_type.r * dt * current_time)
                ) * self.option_type.payoff(paths[episode, current_time])
                delta_v = (
                    torch.normal(0.0, float(self.myagent.sigma), size=(1,))
                    + Q_continuing
                    - r_stopping
                )

                if current_time == n_simulation - 1 or delta_v < 0:
                    continue_ = False
                else:
                    self.myagent.add_to_replay_buffer_continuing(
                        torch.tensor(
                            [current_time, paths[episode, current_time]],
                            dtype=torch.double,
                        ),
                        torch.tensor(
                            [(current_time + 1), paths[episode, current_time + 1]],
                            dtype=torch.double,
                        ),
                        r_stopping,
                    )
                current_time += 1
                history_reward_train.append(r_stopping)
            if n_episodes > start_training:
                history_losses_train[episode] = self.myagent.train_continuing_network()
            history_sigma[episode] = self.myagent.sigma
            self.myagent.sigma = max(
                self.myagent.sigma - self.myagent.sigma_decay,
                self.myagent.sigma_min,
            )
            history_pos_train[episode] = current_time
            if count_target_update % self.myagent.n_update == 0:
                self.myagent.update_target_network()
            count_target_update += 1

        return (
            self.myagent,
            history_losses_train,
            history_reward_train,
            history_pos_train,
            history_sigma,
        )

    def get_price(
        self, trained_agent: agent.Agent, n_simulation: int, n_mc: int
    ) -> float:
        """Return Monte-Carlo
        price of bermudan option with stopping decision given by Q-function.
        Parameters
        ----------
        trained_agent : agent.Agent
            trained agent
        n_simulation : int
            number of timesteps
        n_mc : int
            number of monte-carlo scenarios

        Returns
        -------
        float
            price of bermudan option
        """
        dt = self.option_type.T / n_simulation
        mc_paths = self.process.get_path(n_simulation, dt, n_mc)
        discount_factors = np.exp(-self.option_type.r * dt * np.arange(n_simulation))
        payoffs = np.zeros(n_mc)
        self.myagent.continuing_network.eval()

        for n in tqdm(range(n_mc)):
            current_time = 1
            done = False
            while not done:
                state = torch.tensor(
                    [current_time, mc_paths[n, current_time]], dtype=torch.double
                )
                reward = torch.exp(
                    torch.tensor(-self.option_type.r * dt * current_time)
                ) * self.option_type.payoff(mc_paths[n, current_time])
                Q_continuing = trained_agent.continuing_network(state.unsqueeze(0))
                delta_v = Q_continuing - reward
                if current_time == n_simulation - 1 or delta_v < 0:
                    payoffs[n] = discount_factors[
                        current_time
                    ] * self.option_type.payoff(mc_paths[n, current_time])
                    done = True
                else:
                    current_time += 1

        price = np.mean(payoffs)

        std = np.std(payoffs)  # standard deviation estimator

        error = 1.96 * std / np.sqrt(n_simulation)

        ci_up = price + error
        ci_down = price - error

        return price, ci_up, ci_down
