import sys
import time
from typing import List, Union

sys.path.append("..")

import numpy as np
import torch
from torch import nn
from torch.utils.data import BatchSampler, RandomSampler
from tqdm import tqdm

import src.main.dynamic_programming.dp_pricer as dp_pricer
import src.tools.diffusion.ito_process as ito_process
import src.tools.mlp.mlp as mlp
import src.tools.option.option_main_class as option_main_class


class CustomLoss(nn.Module):
    """Custom Loss for Deep Optimal Stopping

    Parameters
    ----------
    nn : _type_
        _description_
    """

    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(
        self, predicted: torch.tensor, g1: torch.tensor, g2: torch.tensor
    ) -> float:
        """_summary_

        Parameters
        ----------
        predicted : torch.tensor
            prediction of our neural network
        g1 : torch.tensor
            g(n,x_n^k)
        g2 : torch.tensor
            g(l_n,x_{l_n}^k)

        Returns
        -------
        float
        Monte-Carlo loss function as described in Deep Optimal
        Stopping article.

        """
        g1 = torch.squeeze(g1)
        predicted = torch.squeeze(predicted)
        g2 = torch.squeeze(g2)
        loss = torch.mean(g1 * predicted + g2 * (1 - predicted))
        return loss


class PricerDeepOptimalStopping(dp_pricer.PricerDynamicProgramming):
    def __init__(
        self,
        option: option_main_class.Option,
        asset: List[ito_process.ItoProcess],
        weights: List[int],
        learning_rate: float,
        batch_size: int,
        activation_function,
        epochs: int,
    ):
        super().__init__(option, asset)
        self.weights = weights
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.activation_function = activation_function
        self.epochs = epochs
        self.models = []

    def get_strategy(
        self,
        data: np.ndarray,
        current_discount_vect: np.ndarray,
        cash_flows: np.ndarray,
        **kwargs
    ):
        n, d = data.shape
        model_tmp = mlp.NN_DOS(d, 1, self.activation_function, self.weights)
        optimizer = torch.optim.Adam(
            params=model_tmp.parameters(), lr=self.learning_rate, maximize=True
        )
        loss = CustomLoss()
        history_losses = torch.zeros(self.epochs)

        IntrinsicValues = np.squeeze(self.option_type.payoff(data))
        IntrinsicValues = torch.tensor(IntrinsicValues)
        data = torch.tensor(data)
        cash_flows = torch.tensor(cash_flows)

        for epochs in range(self.epochs):
            for batch in range(0, n, self.batch_size):
                inputs = model_tmp(data[batch : batch + self.batch_size, :])
                g1 = IntrinsicValues[batch : batch + self.batch_size]
                g2 = (
                    cash_flows[batch : batch + self.batch_size]
                    * current_discount_vect[batch : batch + self.batch_size]
                )

                losses = loss(inputs, g1, g2)
                history_losses[epochs] = losses.item()

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

        self.models.append(model_tmp)
        model_tmp.eval()
        with torch.no_grad():
            stopping_decisions = torch.argwhere(torch.squeeze(model_tmp(data)) > 0.5)
            stopping_decisions = torch.squeeze(stopping_decisions)

        return stopping_decisions.numpy(), history_losses.numpy()

    def test(self, n, N, seed=None, disable=True):
        start_time = time.time()
        maturity = self.option_type.T
        r = self.option_type.r

        dt = float(maturity / (n - 1))
        d = len(self.asset)

        if d == 1:
            paths = self.asset[0].get_path(n, dt, N, seed)
            paths = np.expand_dims(paths, 2)
        else:
            paths = np.zeros((N, n, d))
            for i in range(d):
                asset_i = self.asset[i]
                paths[:, :, i] = asset_i.get_path(n, dt, N, seed)

        cash_flows = self.option_type.payoff(paths[:, n - 1, :])

        cash_flows = np.squeeze(cash_flows)

        discountVect = np.exp(-r * dt * np.arange(1, n, 1))

        exercice_time = (n - 1) * np.ones(N)
        exercice_time = exercice_time.astype("int")

        loop = tqdm(range(n - 2, 0, -1), disable=disable)
        for step in loop:
            data = paths[:, step, :]
            intrinsic_value = np.squeeze(self.option_type.payoff(data))

            current_model = self.models[n - step - 2]
            current_model.eval()
            with torch.no_grad():
                index_Exercise = torch.argwhere(
                    torch.squeeze(current_model(torch.tensor(data))) > 0.5
                )
                index_Exercise = torch.squeeze(index_Exercise)
            cash_flows[index_Exercise] = intrinsic_value[index_Exercise]
            exercice_time[index_Exercise] = step

        payoff = cash_flows * discountVect[exercice_time - 1]
        price = np.mean(payoff)

        std = np.std(payoff)  # standard deviation estimator

        error = 1.96 * std / np.sqrt(N)

        ci_up = price + error
        ci_down = price - error

        t = time.time() - start_time

        return price, ci_up, ci_down, t, exercice_time
