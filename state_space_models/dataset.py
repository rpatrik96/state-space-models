from torch.utils.data import Dataset
import torch
from .lti import LTISystem


class LTIDataset(Dataset):
    """
    A pytorch dataset that takes an LTI model from lti.py and provides the wrapper
    with the pytorch standard dataset function to provide data for neural nets.
    It takes the number of simulation steps and the variance for the gaussian control signal
    then simulates the data sequence. __getitem__ returns the ith index of the time series
    """

    def __init__(
        self, lti: LTISystem, num_steps: int, variance: float, dt: float = 1e-3
    ):
        self.lti: LTISystem = lti
        self.num_steps: int = num_steps
        self.variance: torch.FloatTensor = torch.FloatTensor([variance])
        self.dt: float = dt
        self.U: torch.Tensor = (
            torch.randn(num_steps, lti.B.shape[1], dtype=torch.float32)
            * self.variance.sqrt()
        )
        self.t, self.y, self.x = lti.simulate(U=self.U.numpy(), dt=self.dt)

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        return self.y[idx]
