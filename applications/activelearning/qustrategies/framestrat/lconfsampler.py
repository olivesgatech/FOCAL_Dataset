import numpy as np
from applications.activelearning.qustrategies.sampler import Sampler
from applications.activelearning.trainer import ALDetectionTrainer
from config import BaseConfig


class FrameLConfSampling(Sampler):
    """
    Class for lconf sampling algorithm. Inherits from sampler.
    """
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig):
        super(FrameLConfSampling, self).__init__(n_pool, start_idxs, cfg)

    def query(self,model, n: int, trainer: ALDetectionTrainer):
        """
        Performs lconf query of points
        """
        stats = trainer.unlabeled_statistics(model)
        frame_lconfs = stats['lconf']

        return np.argsort(frame_lconfs)[-n:]
