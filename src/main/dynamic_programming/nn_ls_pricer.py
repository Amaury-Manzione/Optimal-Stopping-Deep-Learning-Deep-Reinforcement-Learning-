import sys
import time
from typing import List, Union

sys.path.append("..")

import numpy as np
import torch
from torch.utils.data import BatchSampler, RandomSampler
from tqdm import tqdm

import src.main.dynamic_programming.dp_pricer as dp_pricer
import src.tools.diffusion.ito_process as ito_process
import src.tools.mlp.mlp as mlp
import src.tools.option.option_main_class as option_main_class


class PricerNNLongstaffSchwarz(dp_pricer.PricerDynamicProgramming):
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
        self.models = {"underlying": [], "cash_flows": [], "model": []}

    def get_strategy(
        self,
        data: np.ndarray,
        current_discount_vect: np.ndarray,
        cash_flows: np.ndarray,
        intrinsic_value: np.ndarray,
        **kwargs
    ):
        n, d = data.shape
        history_losses = torch.zeros(self.epochs)
        model_tmp = mlp.NN_QDN(d, 1, self.activation_function, self.weights)
        optimizer = torch.optim.Adam(
            params=model_tmp.parameters(), lr=self.learning_rate
        )
        loss = torch.nn.MSELoss()

        ydata = cash_flows * current_discount_vect

        data = torch.tensor(data)
        intrinsic_value = torch.tensor(intrinsic_value)
        ydata = torch.tensor(ydata)

        for epochs in range(self.epochs):
            for batch in range(0, n, self.batch_size):
                inputs = model_tmp(data[batch : batch + self.batch_size, :])
                losses = loss(
                    torch.squeeze(inputs), ydata[batch : batch + self.batch_size]
                )
                history_losses[epochs] = losses.item()

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

        self.models["model"].append(model_tmp)
        model_tmp.eval()
        with torch.no_grad():
            continuation_value = torch.squeeze(model_tmp(data))
            self.models["underlying"].append(torch.squeeze(data))
            self.models["cash_flows"].append(ydata)
            index_Exercise = torch.argwhere(
                torch.squeeze(intrinsic_value) > continuation_value
            )

        return index_Exercise.numpy(), history_losses
