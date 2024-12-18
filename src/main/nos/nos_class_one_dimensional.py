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


class NosOneDimensional:
    """class for Neural Optimal Stopping algorithm"""

    def __init__(
        self,
        asset: List[ito.ItoProcess],
        option_type: option_main_class.Option,
    ):
        """creating object of type NOS

        Parameters
        ----------
        asset : List[ito.ItoProcess]
            underlying for the option
        maturity : float
            maturity of the option
        rate : float
            interest rate
        payoff : Callable
            payoff of the option
        change_coordinates : Callable
            change of coordinates to apply for multi-dimensional asset.
        n_simulation : int
            number of points for discretizing the process
        """
        self.asset = asset
        self.option_type = option_type

    def find_optimal_region(
        self,
        n_simulation: int,
        n_path: int,
        epsilon: float,
        l_girsanov: float,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        activation_function,
        list_weights: List[int],
        steps_lr_scheduler: int,
        gamma=0,
        verbose=True,
    ):
        """_summary_

        Parameters
        ----------
        n_simulation : int
            _description_
        n_path : int
            _description_
        epsilon : float
            _description_
        l_girsanov : float
            _description_
        batch_size : int
            _description_
        learning_rate : float
            _description_
        epochs : int
            _description_
        activation_function : _type_
            _description_
        list_weights : List[int]
            _description_
        steps_lr_scheduler : int
            _description_
        gamma : int, optional
            _description_, by default 0
        verbose : bool, optional
            _description_, by default True

        Returns
        -------
        _type_
            _description_
        """
        t_init = time.time()
        mlp_nos = mlp.NN_NOS(
            len(self.asset),
            1,
            activation_function,
            list_weights,
            last_bias=self.option_type.K / 2,
        )

        # timestep
        dt = float(self.option_type.T / n_simulation)
        timestep = torch.tensor([i * dt for i in range(n_simulation + 1)])

        # history of losses per epoch
        history = torch.zeros(epochs)
        history_var = torch.zeros(epochs)

        # optimizer and scheduler
        optimizer = torch.optim.Adam(
            mlp_nos.parameters(), lr=learning_rate, maximize=True
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=steps_lr_scheduler, gamma=gamma
        )

        # loading the paths
        paths, Z = self.asset[0].get_path_importance_sampling(
            n_simulation + 1, dt, n_path, l=l_girsanov
        )

        paths, Z = torch.tensor(paths, requires_grad=False), torch.tensor(
            Z, requires_grad=False
        )

        list_indexs = np.array([i for i in range(n_path)])

        for epoch in tqdm(range(epochs), disable=verbose):

            index = np.random.choice(list_indexs, size=batch_size)
            paths_batch = paths[index, :]

            # calculating outputs
            input_ = torch.unsqueeze(timestep, 1)
            output = mlp_nos(
                input_.to(torch.double),
            ).squeeze()

            loss = torch.zeros(len(timestep))

            stopping_budgets = torch.ones(batch_size)

            for idx in range(len(timestep) - 1):
                X = paths_batch[:, idx]

                stopping_probabilities = torch.minimum(
                    torch.maximum(
                        (output[idx] - X + epsilon) / (2 * epsilon), torch.tensor(0)
                    ),
                    torch.tensor(1),
                )

                new_stop_budg = stopping_budgets * (1 - stopping_probabilities)

                payoff = self.option_type.payoff(X) * torch.exp(
                    -self.option_type.r * timestep[idx]
                )

                loss[idx] = torch.sum(
                    stopping_probabilities * stopping_budgets * payoff * Z[index, idx]
                )

                stopping_budgets = new_stop_budg

            payoff = self.option_type.payoff(paths_batch[:, -1]) * torch.exp(
                torch.tensor(-self.option_type.r * self.option_type.T)
            )

            stopping_probabilities = torch.ones(batch_size)

            loss[-1] = torch.sum(
                stopping_probabilities * stopping_budgets * payoff * Z[index, -1]
            )

            loss_mean = torch.sum(loss) / batch_size
            loss_std = torch.sum((loss - loss_mean) ** 2) / batch_size

            history[epoch] = loss_mean.item()
            history_var[epoch] = (1.96 * np.sqrt(loss_std.item())) / np.sqrt(batch_size)

            optimizer.zero_grad()
            loss_mean.backward()

            # for name, param in self.mlp.named_parameters():
            #     if param.requires_grad:
            #         print(f"Gradient of {name}:", param.grad)

            optimizer.step()
            scheduler.step()

        t = time.time() - t_init

        return mlp_nos, history, history_var, t

    def get_price(
        self, sharp_region, epsilon, n_simulation: int, n_mc: int, verbose=True
    ):
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
        dt = float(self.option_type.T / (n_simulation - 1))
        timesteps = torch.arange(0, n_simulation, dtype=torch.double) * dt

        # Get sharp regions using batch processing on the entire `timesteps` tensor
        with torch.no_grad():
            sharp_regions = sharp_region(
                timesteps.unsqueeze(1)
            )  # Batch input to the neural network

        # Get asset paths
        paths = torch.tensor(
            self.asset[0].get_path(n_simulation, dt, n_mc), dtype=torch.double
        )

        # Initialize price tensor
        price = torch.zeros(n_mc, dtype=torch.double)

        # Vectorize discount factor computation
        discount_factors = torch.exp(
            -self.option_type.r * dt * torch.arange(n_simulation, dtype=torch.double)
        )

        # Vectorized Monte Carlo path processing
        for n in tqdm(range(n_mc), disable=verbose):
            # Initialize index for stopping condition
            idx = -1

            # Check for stopping condition at each timestep
            for i in range(n_simulation):
                if sharp_regions[i] - paths[n, i] >= epsilon:
                    idx = i
                    break  # Exit the loop on first condition met

            # Compute the discounted payoff based on the found index
            if idx != -1:
                price[n] = (
                    self.option_type.payoff(paths[n, idx]) * discount_factors[idx]
                )
            else:
                price[n] = self.option_type.payoff(paths[n, -1]) * discount_factors[-1]

        mean = torch.mean(price)
        error = (1.96 * torch.std(price)) / torch.sqrt(n_simulation)

        ci_up = mean + error
        ci_down = mean - error

        # Return the average price across Monte Carlo simulations
        return mean, ci_up, ci_down
