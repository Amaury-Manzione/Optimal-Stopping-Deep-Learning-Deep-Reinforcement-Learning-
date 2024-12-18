import sys
import time
from typing import Callable, List, Union

sys.path.append("..")

import numpy as np
import torch
from tqdm import tqdm

import src.main.pricer as pricer
import src.tools.diffusion.ito_process as ito_process
import src.tools.mlp.mlp as mlp
import src.tools.option.option_main_class as option_main_class


class NOS(pricer.Pricer):
    def __init__(
        self,
        option_: option_main_class.Option,
        asset_: List[ito_process.ItoProcess],
        change_coordinates: Callable,
        epsilon: float,
        l_girsanov: float,
        weights: List[int],
        last_bias: float,
        activation_function,
        learning_rate: float,
        epochs: int,
        batch_size: int,
    ):
        super().__init__(option_, asset_)
        # nos parameters
        self.change_coordinates = change_coordinates
        self.epsilon = epsilon
        self.l_girsanov = l_girsanov

        # nn parameters
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        # defining neural network and optimizer
        input_dim = 1 if len(asset_) == 1 else len(asset_) + 1
        self.nn = mlp.NN_NOS(input_dim, 1, activation_function, weights, last_bias)
        self.optimizer = torch.optim.Adam(
            self.nn.parameters(), lr=learning_rate, maximize=True
        )

    def train_one_dim(self, data, Z, timestep):
        # calculating outputs
        input_ = torch.unsqueeze(timestep, 1)
        output = self.nn(
            input_.to(torch.double),
        ).squeeze()

        loss = torch.zeros(len(timestep))

        stopping_budgets = torch.ones(self.batch_size)

        for idx in range(len(timestep) - 1):
            X = torch.squeeze(data[:, idx, :])

            stopping_probabilities = torch.minimum(
                torch.maximum(
                    (output[idx] - X + self.epsilon) / (2 * self.epsilon),
                    torch.tensor(0),
                ),
                torch.tensor(1),
            )

            new_stop_budg = stopping_budgets * (1 - stopping_probabilities)

            payoff = self.option_type.payoff(X) * torch.exp(
                -self.option_type.r * timestep[idx]
            )

            loss[idx] = torch.sum(
                stopping_probabilities * stopping_budgets * payoff * Z[:, idx]
            )

            stopping_budgets = new_stop_budg

        return stopping_probabilities, stopping_budgets, loss

    def train_multi_dim(self, data, Z, timestep):
        loss = torch.zeros(len(timestep))

        stopping_budgets = torch.ones(self.batch_size)

        for idx in range(len(timestep) - 1):
            tensors = []
            time_tensor = timestep[idx] * torch.ones(self.batch_size, dtype=torch.float)

            tensors.append(time_tensor)

            X = data[:, idx, :]
            alpha_X = self.change_coordinates(torch.squeeze(X))
            for dim in range(len(self.asset)):
                tensors.append(X[:, dim] / alpha_X)

            input_ = torch.stack(tensors, dim=1)

            output = self.nn(
                input_,
            ).squeeze()

            stopping_probabilities = torch.minimum(
                torch.maximum(
                    (-output + alpha_X + self.epsilon) / (2 * self.epsilon),
                    torch.tensor(0),
                ),
                torch.tensor(1),
            )

            new_stop_budg = stopping_budgets * (1 - stopping_probabilities)

            payoff = self.option_type.payoff(torch.squeeze(X))
            payoff = payoff * torch.exp(-self.option_type.r * timestep[idx])

            loss[idx] = torch.sum(
                stopping_probabilities * stopping_budgets * payoff * Z[:, idx]
            )

            stopping_budgets = new_stop_budg

        return stopping_probabilities, stopping_budgets, loss

    def find_optimal_region(
        self,
        n_simulation: int,
        n_path: int,
        disable=True,
    ):
        t_init = time.time()
        d = len(self.asset)

        # timestep
        dt = float(self.option_type.T / n_simulation)
        timestep = torch.tensor([i * dt for i in range(n_simulation + 1)])

        # history of losses per epoch
        history = torch.zeros(self.epochs)
        history_var = torch.zeros(self.epochs)

        # loading the paths
        paths, Z = np.zeros((n_path, n_simulation + 1, d)), np.ones(
            (n_path, n_simulation + 1)
        )
        for i in range(d):
            paths[:, :, i], Z_tmp = self.asset[i].get_path_importance_sampling(
                n_simulation + 1, dt, n_path, l=self.l_girsanov
            )
            Z *= Z_tmp

        paths, Z = torch.tensor(paths, requires_grad=False), torch.tensor(
            Z, requires_grad=False
        )
        Z = torch.exp(-0.5 * torch.tensor(self.l_girsanov**2) * timestep) * Z

        list_indexs = np.array([i for i in range(n_path)])

        loop = tqdm(range(self.epochs), disable=disable)
        for epoch in loop:
            index = np.random.choice(list_indexs, size=self.batch_size)
            paths_batch = paths[index, :, :]
            Z_batch = Z[index, :]

            stopping_probabilities, stopping_budgets, loss = (
                self.train_one_dim(paths_batch, Z_batch, timestep)
                if d == 1
                else self.train_multi_dim(paths_batch, Z_batch, timestep)
            )
            payoff = self.option_type.payoff(
                torch.squeeze(paths_batch[:, -1, :])
            ) * torch.exp(torch.tensor(-self.option_type.r * self.option_type.T))

            stopping_probabilities = torch.ones(self.batch_size)

            loss[-1] = torch.sum(
                stopping_probabilities * stopping_budgets * payoff * Z_batch[:, -1]
            )

            loss_mean = torch.sum(loss) / self.batch_size
            loss_std = torch.sum((loss - loss_mean) ** 2) / self.batch_size

            history[epoch] = loss_mean.item()
            history_var[epoch] = (1.96 * np.sqrt(loss_std.item())) / np.sqrt(
                self.batch_size
            )

            self.optimizer.zero_grad()
            loss_mean.backward()
            self.optimizer.step()

            loop.set_postfix(loss=loss_mean.item())

        t = time.time() - t_init

        return history, history_var, t

    def price(self, n: int, N: int, **kwargs):
        pass
