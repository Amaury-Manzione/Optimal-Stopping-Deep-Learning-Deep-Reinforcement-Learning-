import sys
import time
from typing import List

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

sys.path.append("..\..")


class NN_DOS(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation_function,
        list_weights: List[int],
    ):
        """Construct Multi-Layer Perceptron for Deep Optimal Stopping Algorithm,
        requires Sigmoid activation function for the last layer and bacth normalization
        Parameters
        ----------
        input_dim : int
            dimension of input vector
        output_dim : int
            dimension of output vector
        activation_function : _type_
            activation function to use for hidden layers
        list_weights : List[int]
            number of weights for each layer
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.list_weights = list_weights

        layers = []
        layers.append(nn.Linear(input_dim, list_weights[0]))
        layers.append(nn.BatchNorm1d(list_weights[0]))
        layers.append(activation_function)

        # Add additional hidden layers
        num_layers = len(self.list_weights)
        for i in range(num_layers - 1):
            layers.append(nn.Linear(list_weights[i], list_weights[i + 1]))
            layers.append(nn.BatchNorm1d(list_weights[i + 1]))
            layers.append(activation_function)

        layers.append(nn.Linear(list_weights[num_layers - 1], output_dim))
        layers.append(nn.Sigmoid())

        self.linear_relu_stack = nn.Sequential(*layers)
        self.double()

    def forward(self, x):
        """Forward  pass

        Parameters
        ----------
        x : _type_
            input tensor

        Returns
        -------
        _type_
            _description
        """
        output = self.linear_relu_stack(x)
        return output


class NN_NOS(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation_function,
        list_weights: List[int],
        last_bias: int,
    ):
        """Construct Multi-Layer Perceptron for Neural Optimal Stopping Algorithm,
        requires initialization of the last biais for the last layer
        and bacth normalization
        Parameters
        ----------
        input_dim : int
            dimension of input vector
        output_dim : int
            dimension of output vector
        activation_function : _type_
            activation function to use for hidden layers
        list_weights : List[int]
            number of weights for each layer
        last_bias : int
            value of last biais
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.list_weights = list_weights
        self.last_bias = last_bias

        layers = []
        layers.append(nn.Linear(input_dim, list_weights[0]))
        layers.append(nn.BatchNorm1d(list_weights[0]))
        layers.append(activation_function)

        # Add additional hidden layers
        num_layers = len(self.list_weights)
        for i in range(num_layers - 1):
            layers.append(nn.Linear(list_weights[i], list_weights[i + 1]))
            layers.append(nn.BatchNorm1d(list_weights[i + 1]))
            layers.append(activation_function)

        layers.append(nn.Linear(list_weights[num_layers - 1], output_dim))

        self.linear_relu_stack = nn.Sequential(*layers)

        self.init_weights(last_bias)
        self.double()

    def init_weights(self, last_bias: int):
        """initialize weights with xavier algorithm and last bias of mlp.

        Parameters
        ----------
        last_bias : int
            _description_
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Initialize linear layer weights using Xavier initialization
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(self.linear_relu_stack[-1], nn.Linear):
                    torch.nn.init.constant_(self.linear_relu_stack[-1].bias, last_bias)

    def forward(self, x):
        """Forward  pass

        Parameters
        ----------
        x : _type_
            input tensor

        Returns
        -------
        _type_
            _description
        """
        output = self.linear_relu_stack(x)
        return output


class NN_QDN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation_function,
        list_weights: List[int],
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.list_weights = list_weights

        layers = []
        layers.append(nn.Linear(input_dim, list_weights[0]))
        layers.append(nn.BatchNorm1d(list_weights[0]))
        layers.append(activation_function)

        # Add additional hidden layers
        num_layers = len(self.list_weights)
        for i in range(num_layers - 1):
            layers.append(nn.Linear(list_weights[i], list_weights[i + 1]))
            layers.append(nn.BatchNorm1d(list_weights[i + 1]))
            layers.append(activation_function)

        layers.append(nn.Linear(list_weights[num_layers - 1], output_dim))

        self.linear_relu_stack = nn.Sequential(*layers)
        self.double()

    def forward(self, x):
        """Forward  pass

        Parameters
        ----------
        x : _type_
            input tensor

        Returns
        -------
        _type_
            _description
        """
        output = self.linear_relu_stack(x)
        return output
