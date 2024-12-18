from typing import List

import numpy as np
import torch
from tensordict import TensorDict
from torch import nn
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer

import src.tools.mlp.mlp as mlp


class Agent:
    """class representing a dqn agent."""

    def __init__(
        self,
        epsilon: float,
        epsilon_decay: float,
        epsilon_max: float,
        replay_memory_int: int,
        replay_memory_capacity: int,
        n_update: int,
        learning_rate: float,
        batch_size: int,
        activation_function,
        list_weights: List[int],
        d: float,
    ):
        """Object representing the agent

        Parameters
        ----------
        epsilon : float
            value for epsilon-greedy exploration
        epsilon_decay : float
            value for deacreasing epsilon
        epsilon_max : float
            maximum value for epsilon
        replay_memory_init : int
            minimum value for sampling in the replay buffer
        replay_memory_capacity : int
            maximum number of samples in the replay buffer
        n_update : int
            change target network parameters every n_update iteration
        learning_rate : float
            learning rate for training the online network
        batch_size : int
            number of batches for backpropagation
        activation_function
            activation function for multi-layer perceptron
        d : float
            dimension of state space
        """
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_max = epsilon_max
        self.replay_memory_init = replay_memory_int
        self.replay_memory_capacity = replay_memory_capacity
        self.n_update = n_update
        self.batch_size = batch_size

        # replay buffer
        storage = LazyMemmapStorage(replay_memory_capacity)
        replay_buffer = TensorDictReplayBuffer(storage=storage, batch_size=batch_size)
        self.replay_buffer = replay_buffer

        # neural networks
        self.online_network = mlp.NN_QDN(d + 1, 2, activation_function, list_weights)
        self.target_network = mlp.NN_QDN(d + 1, 2, activation_function, list_weights)
        for p in self.target_network.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=learning_rate, maximize=False
        )
        self.criterion = nn.MSELoss()

    def get_Q(self, state: torch.tensor) -> torch.tensor:
        """
        Parameters
        ----------
        state : torch.tensor
            current state

        Returns
        -------
        torch.tensor
            Q function for all possible actions
        """
        self.online_network.eval()
        list_actions = self.online_network(state.unsqueeze(0))
        return list_actions

    def get_policy(self, state) -> int:
        """
        return optimal policy ie argmax over all possible actions on Q-function
        Parameters
        ----------
        state : torch.tensor
            current_state

        Returns
        -------
        int
            best action given current state
        """
        list_actions = self.get_Q(state)
        action = torch.argmax(list_actions).item()
        return action

    # def epsilon_greedy(self, state: torch.tensor, p: int) -> bool:
    #     u = np.random.uniform(0, 1)
    #     if u > self.epsilon:
    #         v = np.random.uniform(0, 1)
    #         action = 1 if v < p else 0
    #     else:
    #         action = self.get_policy(state)

    #     self.epsilon = min(self.epsilon + self.epsilon_decay, self.epsilon_max)
    #     return action

    def train_online_network(
        self,
    ) -> tuple[float, float]:
        """
        One epoch of online network training with Bellmann Loss.
        Returns
        -------
        tuple[float, float]
            loss and mean reward per batch
        """

        if len(self.replay_buffer) < self.replay_memory_init:
            return 0, 0

        else:
            self.online_network.train()
            # calculating Q values
            batch = self.replay_buffer.sample()
            outputs = self.online_network(batch["state"])
            outputs = torch.squeeze(outputs)

            list_actions = batch["action"].numpy()
            outputs_loss = torch.zeros(self.batch_size)
            for i in range(self.batch_size):
                outputs_loss[i] = outputs[i, int(list_actions[i])]

            # calculating loss function
            target = self.target_network(batch["next_state"])
            target = torch.squeeze(target)
            target = batch["reward"] + (
                torch.max(target, dim=1)[0] * (1 - batch["done"])
            )

            loss = self.criterion(outputs_loss.double(), target.double())

            history_loss = loss.item()
            non_zero_mask = batch["done"] == True
            non_zero_rewards = batch["reward"].masked_select(non_zero_mask)
            history_reward = torch.mean(non_zero_rewards)

            # stopped_pos = batch["pos"].masked_select(non_zero_mask)
            # if torch.mean(stopped_pos) > 25:
            #     loss = loss + 10

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return history_loss, float(history_reward)

    def update_target_network(self):
        """
        copy online network parameters to the target  network
        """
        self.target_network.load_state_dict(self.online_network.state_dict())

    def add_to_replay_buffer(
        self,
        current_state: torch.tensor,
        new_state: torch.tensor,
        reward: torch.tensor,
        action: torch.tensor,
        done: torch.tensor,
        pos: torch.tensor,
    ):
        """Add experience to replay buffer

        Parameters
        ----------
        current_state : torch.tensor
        new_state : torch.tensor
        reward : torch.tensor
        action : torch.tensor
        done : torch.tensor
        pos : torch.tensor
        """
        self.replay_buffer.add(
            TensorDict(
                {
                    "state": current_state,
                    "next_state": new_state,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "pos": pos,
                },
                batch_size=[],
            )
        )
