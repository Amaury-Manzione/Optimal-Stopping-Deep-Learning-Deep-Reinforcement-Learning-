from abc import ABCMeta, abstractmethod
from typing import List

import src.tools.diffusion.ito_process as ito_process
import src.tools.option.option_main_class as option_main_class


class Pricer(metaclass=ABCMeta):
    """Combine an object of type option and an object type ito_process to price
     a bermudan option.

    Parameters
    ----------
    metaclass : _type_, optional
        _description_, by default ABCMeta
    """

    def __init__(
        self, option_: option_main_class.Option, asset_: List[ito_process.ItoProcess]
    ):
        self.option_type = option_
        self.asset = asset_

    @abstractmethod
    def price(self, n: int, N: int, **kwargs):
        """
        Parameters
        ----------
        n : int
            number of timesteps
        N : int
            number of Monte-Carlo scenarios
        Raises
        ------
        Exception
            _description_
        """
        raise Exception(
            "price() method is not implemented in one of the subclasses of Pricer"
        )
