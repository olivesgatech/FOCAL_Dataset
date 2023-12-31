import numpy as np
from applications.activelearning.qustrategies.sampler import Sampler
from config import BaseConfig


class FrameRandomSampling(Sampler):
    """
    Class for random sampling algorithm. Inherits from sampler.
    """
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig):
        super(FrameRandomSampling, self).__init__(n_pool, start_idxs, cfg)

    def query(self, n: int, trainer):
        """
        Performs random query of points
        """
        inds = np.where(self.total_pool == 0)[0]
        return inds[np.random.permutation(len(inds))][:n]
