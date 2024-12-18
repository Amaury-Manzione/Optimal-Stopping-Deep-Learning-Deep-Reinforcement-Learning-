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
        sigma: float,
        sigma_decay: float,
        sigma_min: float,
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
        sigma : float
            value for sigma-greedy exploration
        sigma_decay : float
            value for deacreasing sigma
        sigma_min : float
            maximum value for sigma
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
        list_weights : List[int]
            weights for mlp
        d : float
            dimension of state space
        """
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.sigma_min = sigma_min
        self.replay_memory_init = replay_memory_int
        self.replay_memory_capacity = replay_memory_capacity
        self.n_update = n_update
        self.batch_size = batch_size

        # replay buffer
        storage_continuing = LazyMemmapStorage(replay_memory_capacity)
        replay_buffer_continuing = TensorDictReplayBuffer(
            storage=storage_continuing, batch_size=batch_size
        )
        self.replay_buffer_continuing = replay_buffer_continuing

        # neural networks
        self.continuing_network = mlp.NN_QDN(
            d + 1,
            1,
            activation_function,
            list_weights,
        )

        self.target_network = mlp.NN_QDN(
            d + 1,
            1,
            activation_function,
            list_weights,
        )
        for p in self.target_network.parameters():
            p.requires_grad = False

        # optimizers
        self.optimizer = torch.optim.Adam(
            self.continuing_network.parameters(), lr=learning_rate
        )

        self.criterion = nn.MSELoss()

    def train_continuing_network(
        self,
    ) -> tuple[float, float]:
        """
        One epoch of online network training with Bellmann Loss.
        Returns
        -------
        tuple[float, float]
            loss and mean reward per batch
        """
        self.continuing_network.train()
        if len(self.replay_buffer_continuing) < self.replay_memory_init:
            return 0

        else:
            # calculating Q values
            batch = self.replay_buffer_continuing.sample()
            outputs = self.continuing_network(batch["state"])
            outputs = torch.squeeze(outputs)

            # calculating loss function
            target_c = self.target_network(batch["next_state"])
            target_c = torch.squeeze(target_c)

            target_s = batch["reward"]

            target = torch.max(target_c, target_s)

            loss = self.criterion(outputs.double(), target.double())

            # history_loss = loss.item()
            # non_zero_mask = batch["done"] == True
            # non_zero_rewards = batch["reward"].masked_select(non_zero_mask)
            # history_reward = torch.mean(non_zero_rewards)

            # stopped_pos = batch["pos"].masked_select(non_zero_mask)
            # if torch.mean(stopped_pos) > 25:
            #     loss = loss + 10

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """
        copy online network parameters to the target  network
        """
        self.target_network.load_state_dict(self.continuing_network.state_dict())

    def add_to_replay_buffer_continuing(
        self, current_state: torch.tensor, new_state: torch.tensor, reward: torch.tensor
    ):
        """Add experience to replay buffer

        Parameters
        ----------
        current_state : torch.tensor
        new_state : torch.tensor
        reward : torch.tensor
        """
        self.replay_buffer_continuing.add(
            TensorDict(
                {"state": current_state, "next_state": new_state, "reward": reward},
                batch_size=[],
            )
        )

    def get_policy(self, current_reward: torch.tensor, input_tensor: torch.tensor):
        Q_continuing = self.continuing_network(input_tensor.unsqueeze(0))
        return int(torch.squeeze(Q_continuing) - current_reward > 0)
