import sys
import time
from typing import Callable, List

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

sys.path.append("..\..")

import src.tools.diffusion.ito_process as ito
import src.tools.mlp.mlp as mlp
import src.tools.option.option_main_class as option_main_class

# torch.autograd.set_detect_anomaly(True)


class NosMultiDimensional:
    """class for Neural Optimal Stopping algorithm"""

    def __init__(
        self,
        asset: List[ito.ItoProcess],
        option_type: option_main_class.Option,
        change_coordinates: Callable,
    ):
        """creating object of type NOS

        Parameters
        ----------
        asset : List[ito.ItoProcess]
            underlying for the option
        T : float
            T of the option
        r : float
            interest r
        payoff : Callable
            payoff of the option
        change_coordinates : Callable
            change of coordinates to apply for multi-dimensional asset.
        n_simulation : int
            number of points for discretizing the process
        """
        self.asset = asset
        self.option_type = option_type
        self.change_coordinates = change_coordinates

    def find_optimal_region(
        self,
        n_simulation: int,
        n_path: int,
        epsilon: float,
        l_girsanov: float,
        batch_size: int,
        learning_r: float,
        epochs: int,
        activation_function: int,
        weights,
        verbose=True,
    ):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        # dimension
        d = len(self.asset)

        t_init = time.time()
        mlp_nos = mlp.NN_NOS(
            d + 1,
            1,
            activation_function,
            weights,
            last_bias=self.option_type.K,
        )

        # timestep
        dt = float(self.option_type.T / (n_simulation - 1))
        timestep = torch.tensor([i * dt for i in range(n_simulation)])

        # history of losses per epoch
        history = torch.zeros(epochs)

        # optimizer and scheduler
        optimizer = torch.optim.Adam(mlp_nos.parameters(), lr=learning_r, maximize=True)
        # learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer,
        #     step_size=100,
        #     gamma=1e-6 / 1e-2,
        # )

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.02)
        paths, Z = self.asset[0].get_path_importance_sampling_test(
            d, n_simulation, dt, n_path, l_girsanov, d
        )
        paths, Z = torch.tensor(paths, requires_grad=False, dtype=float), torch.tensor(
            Z, requires_grad=False, dtype=float
        )

        list_indexs = np.array([i for i in range(n_path)])

        # calculation of epsilon
        # epsilon_array = torch.zeros(n_simulation)
        # for step in range(n_simulation):
        #     X_tmp = paths[:, step, :]
        #     alpha_X = torch.tensor(
        #         [min(row) / max(row) for row in X_tmp], dtype=torch.float
        #     )
        #     epsilon_array[step] = 110 * (torch.std(alpha_X)) * np.sqrt(dt) * 0.2
        #     print(f"epsilon is {epsilon_array[step]}")

        for epoch in tqdm(range(epochs), disable=verbose):

            index = np.random.choice(list_indexs, size=batch_size)
            paths_batch = paths[index, :, :]

            loss = torch.tensor(0.0)

            stopping_budgets = torch.ones(batch_size)

            for idx in range(len(timestep) - 1):
                tensors = []

                time_tensor = timestep[idx] * torch.ones(batch_size, dtype=torch.float)

                tensors.append(time_tensor)

                X = torch.squeeze(paths_batch[:, idx, :])
                alpha_X = torch.tensor(
                    [self.change_coordinates(row) for row in X],
                    dtype=torch.float,
                )

                for i in range(d):
                    tensors.append(X[:, i] / alpha_X)

                input_ = torch.stack(tensors, dim=1)
                # input_ = torch.squeeze(input_)

                output = mlp_nos(
                    input_,
                ).squeeze()

                stopping_probabilities = torch.minimum(
                    torch.maximum(
                        (-output + alpha_X + epsilon) / (2 * epsilon),
                        torch.tensor(0),
                    ),
                    torch.tensor(1),
                )

                new_stop_budg = stopping_budgets * (1 - stopping_probabilities)

                payoff = torch.tensor(
                    [self.option_type.payoff(row) for row in X], dtype=torch.float
                )
                payoff = payoff * torch.exp(-self.option_type.r * timestep[idx])

                loss = loss + torch.sum(
                    stopping_probabilities * stopping_budgets * payoff * Z[index, idx]
                )

                stopping_budgets = new_stop_budg

            payoff = torch.tensor(
                [
                    self.option_type.payoff(row)
                    for row in torch.squeeze(paths_batch[:, -1, :])
                ],
                dtype=torch.float,
            )
            payoff = payoff * torch.exp(
                torch.tensor(-self.option_type.r * self.option_type.T)
            )

            stopping_probabilities = torch.ones(batch_size)

            loss = loss + torch.sum(
                stopping_probabilities * stopping_budgets * payoff * Z[index, -1]
            )

            loss = loss / batch_size

            history[epoch] = loss.item()
            # print(loss.item())

            optimizer.zero_grad()
            loss.backward()

            # for name, param in mlp_nos.named_parameters():
            #     if param.requires_grad:
            #         print(f"Gradient of {name}:", param.grad)

            optimizer.step()
            # learning_rate_scheduler.step()

        t = time.time() - t_init

        return mlp_nos, history, t

    def get_price(self, sharp_region, n_simulation: int, n_mc: int, verbose=True):
        """Get price of bermudan option given the trained region.

        Parameters
        ----------
        sharp_region : _type_
            trained stop region.
        n_simulation : int
            number of points for discretizing the process.
        n_mc : int
            number of monte-carlo scenarios.
        """
        # timestep
        dt = float(self.option_type.T / n_simulation)
        timesteps = torch.tensor([i * dt for i in range(n_simulation)])

        sharp_regions = sharp_region(torch.unsqueeze(timesteps, 1))

        paths = self.option_type.asset[0].get_path(n_simulation, dt, n_mc)
        paths = torch.tensor(paths)

        price = torch.zeros(n_mc)

        for n in tqdm(range(n_mc), disable=verbose):
            for idx in range(n_simulation):
                if paths[n, idx] <= sharp_regions[idx]:
                    price[n] = self.option_type.option_type.payoff(
                        paths[n, idx]
                    ) * np.exp(-self.option_type.r * dt * idx)
                    break
            price[n] = self.option_type.payoff(paths[n, idx]) * np.exp(
                -self.option_type.r * dt * idx
            )

        return torch.mean(price)
