from typing import Callable


class Option:
    """
    Class representing an option ( we treat the cases of european, american, maxoption)
    """

    def __init__(self, T: float, K: float, r: float, option_payoff: Callable):
        """
        Parameters
        ----------
        T : float
            maturity
        K : float
            strike
        r : float
            interest rate
        option_payoff : Callable
            type of the option (european,american or maxoption)
        """
        self.T = T
        self.K = K
        self.r = r
        self.payoff = option_payoff

    # def payoff_scalar(self, t: int, asset: np.array) -> np.array:
    #     """_summary_

    #     Parameters
    #     ----------
    #     t : int
    #         time between O and T
    #     asset : np.array
    #         underlying

    #     Returns
    #     -------
    #     np.array
    #         returns payoff of the option when the asset is numpy array.

    #     Raises
    #     ------
    #     NameError
    #         _description_
    #     """

    #     return self.payoff(asset[:, t, :])

    # def payoff_tensor(self, t: int, asset: torch.tensor) -> torch.tensor:
    #     """_summary_

    #     Parameters
    #     ----------
    #     t : int
    #         time between 0 and T
    #     asset : torch.tensor
    #         underlying

    #     Returns
    #     -------
    #     torch.tensor
    #         returns payoff of the option when the asset is a tensor
    #         (for DOS algorithm mainly).

    #     Raises
    #     ------
    #     NameError
    #         _description_
    #     """
    #     match self.option_type:
    #         case "american":
    #             asset = torch.squeeze(asset)
    #             if self.is_call_or_put == "call":
    #                 return torch.maximum(asset[:, t] - self.K, torch.tensor(0))
    #             else:
    #                 return torch.maximum(self.K - asset[:, t], torch.tensor(0))
    #         case "maxoption":
    #             if self.is_call_or_put == "call":
    #                 payoff_ = torch.maximum(
    #                     torch.max(asset[:, t, :], dim=1).values - self.K,
    #                     torch.tensor(0),
    #                 )
    #             else:
    #                 payoff_ = torch.maximum(
    #                     self.K - torch.max(asset[:, t, :], dim=1).values,
    #                     torch.tensor(0),
    #                 )
    #             return payoff_
    #         case _:
    #             raise NameError("incorrect name for option type.")
