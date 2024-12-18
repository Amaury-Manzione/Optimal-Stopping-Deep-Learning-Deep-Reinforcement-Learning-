import sys
import time as time
from typing import Callable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.append("..\..")

import src.main.dqn.agent as agent
import src.tools.diffusion.ito_process as ito_process
import src.tools.option.option_main_class as option_main_class


class dqn:
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

    def explore(
        self, episode: int, list_times: torch.tensor, paths: torch.tensor, dt: float
    ) -> int:
        """epsilon-greedy exploration : agent choose a stopping time randomly and
        experience to buffer.
        Parameters
        ----------
        episode : int
            current episode
        list_times : torch.tensor
            list of times to sample stopping time
        paths : torch.tensor
            tensor of paths
        dt : float
            timestep

        Returns
        -------
        int
            sampled stopping time
        """
        stopping_time_index = torch.multinomial(
            torch.ones(list_times.size()), 1, replacement=True
        )
        stopping_time = list_times[stopping_time_index].item()
        reward = torch.exp(
            torch.tensor(-self.option_type.r * dt * stopping_time)
        ) * self.option_type.payoff(paths[episode, stopping_time])
        self.myagent.add_to_replay_buffer(
            torch.tensor(
                [paths[episode, stopping_time], stopping_time * dt], dtype=torch.double
            ),
            torch.tensor(
                [paths[episode, stopping_time], stopping_time * dt], dtype=torch.double
            ),
            reward,
            torch.tensor(0),
            torch.tensor(1),
            torch.tensor(stopping_time, dtype=torch.float),
        )
        for i in range(stopping_time - 1):
            self.myagent.add_to_replay_buffer(
                torch.tensor([paths[episode, i], i * dt]),
                torch.tensor([paths[episode, i + 1], (i + 1) * dt]),
                torch.tensor(0),
                torch.tensor(1),
                torch.tensor(0),
                torch.tensor(i, dtype=torch.float),
            )

        return stopping_time

    def exploit(self, episode: int, paths: torch.tensor, dt: float) -> tuple[int, int]:
        """Exploitation of parametrized Q-function : action chosen is the one that
        maximizes Q-function.

        Parameters
        ----------
        episode : int
            current episode
        paths : torch.tensor
            tensor of paths
        dt : _type_
            timestep

        Returns
        -------
        tuple[int, int]
            stopping time and terminal reward
        """
        done = False
        current_time = 1
        state = paths[episode, current_time]
        n = paths.shape[1]
        reward = torch.tensor(0)
        while not done:
            state_time = torch.tensor([state, current_time * dt], dtype=torch.double)
            action = self.myagent.get_policy(state_time)
            if action == 0 or current_time == n - 1:
                reward = torch.exp(
                    torch.tensor(-self.option_type.r * dt * current_time)
                ) * self.option_type.payoff(state)
                done = True
                action = 0
                self.myagent.add_to_replay_buffer(
                    torch.tensor(
                        [state, current_time * dt],
                        dtype=torch.double,
                    ),
                    torch.tensor(
                        [state, current_time * dt],
                        dtype=torch.double,
                    ),
                    reward,
                    torch.tensor(action),
                    torch.tensor(int(done)),
                    torch.tensor(current_time, dtype=torch.float),
                )
            else:
                new_state = paths[episode, current_time + 1]
                self.myagent.add_to_replay_buffer(
                    torch.tensor(
                        [state, current_time * dt],
                        dtype=torch.double,
                    ),
                    torch.tensor(
                        [new_state, (current_time + 1) * dt],
                        dtype=torch.double,
                    ),
                    reward,
                    torch.tensor(action),
                    torch.tensor(int(done)),
                    torch.tensor(current_time, dtype=torch.float),
                )
                current_time += 1
                state = new_state
        return current_time, reward

    def train(
        self,
        n_simulation: int,
        n_episodes: int,
        start_of_training,
        verbose=False,
        disable=True,
        seed=None,
    ) -> tuple[
        torch.tensor,
        torch.tensor,
        torch.tensor,
        torch.tensor,
        torch.tensor,
        agent.Agent,
    ]:
        """training dqn agent.

        Parameters
        ----------
        n_simulation : int
            number of timesteps
        n_episodes : int
            number of episode
        verbose : bool, optional
            display reward and stopping times, by default False
        disable : bool, optional
            allow tqdm or not, by default True
        seed : _type_, optional
            seed for random number generation, by default None

        Returns
        -------
        tuple[ torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, agent.Agent, ]
            reward per episode, reward per batch, loss per episode, epsilon per epoch,
            history of exploitation / exploration, trained agent.
        """
        # timestep
        dt = float(self.option_type.T / (n_simulation - 1))
        list_times = torch.tensor([i for i in range(n_simulation)])

        # loading the environment
        paths = self.process.get_path(n_simulation, dt, n_episodes, seed)
        paths = torch.tensor(paths, dtype=torch.double)

        # history of rewards per episode
        history_batch_reward = torch.zeros(n_episodes)
        history_reward = torch.zeros(n_episodes)
        history_losses = torch.zeros(n_episodes)
        history_pos = torch.zeros(n_episodes)
        history_epsilon = torch.zeros(n_episodes)
        history_explore_exploit = torch.zeros(n_episodes)

        count_target_update = 0

        # training loop
        for episode in tqdm(range(n_episodes), disable=disable):
            # norm_euclid = np.mean(paths[episode, :] ** 2)
            u = np.random.uniform()
            if u > self.myagent.epsilon:
                history_pos[episode] = self.explore(episode, list_times, paths, dt)
                history_explore_exploit[episode] = 0
            else:
                history_pos[episode], history_reward[episode] = self.exploit(
                    episode, paths, dt
                )
                history_explore_exploit[episode] = 1
            if episode > start_of_training:
                history_losses[episode], history_batch_reward[episode] = (
                    self.myagent.train_online_network()
                )
                count_target_update += 1
                if count_target_update % self.myagent.n_update == 0:
                    self.myagent.update_target_network()
                if verbose:
                    print(
                        f"Episode {episode}, Reward: {history_batch_reward[episode]}, pos : {history_pos[episode]}"
                    )

            history_epsilon[episode] = self.myagent.epsilon
            self.myagent.epsilon = min(
                self.myagent.epsilon + self.myagent.epsilon_decay,
                self.myagent.epsilon_max,
            )

        return (
            history_reward,
            history_batch_reward,
            history_losses,
            history_pos,
            history_epsilon,
            history_explore_exploit,
            self.myagent,
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
        dt = float(self.option_type.T / (n_simulation - 1))
        mc_paths = self.process.get_path(n_simulation, dt, n_mc)
        discount_factors = np.exp(-self.option_type.r * dt * np.arange(n_simulation))
        payoffs = np.zeros(n_mc)

        for n in tqdm(range(n_mc)):
            current_time = 1
            done = False
            while not done:
                state = torch.tensor([mc_paths[n, current_time], current_time * dt])
                action = trained_agent.get_policy(state)
                if current_time == n_simulation - 1 or action == 0:
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
