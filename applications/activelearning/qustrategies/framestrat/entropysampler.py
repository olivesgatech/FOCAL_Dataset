import numpy as np
from applications.activelearning.qustrategies.sampler import Sampler
from applications.activelearning.trainer import ALDetectionTrainer
from config import BaseConfig


class FrameEntropySampling(Sampler):
    """
    Class for random sampling algorithm. Inherits from sampler.
    """
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig):
        super(FrameEntropySampling, self).__init__(n_pool, start_idxs, cfg)

    def query(self,model, n: int, trainer: ALDetectionTrainer):
        """
        Performs entropy query of points
        """
        stats = trainer.unlabeled_statistics(model)
        relevant_events = {}
        frame_entropies = stats['entropy']

        return np.argsort(frame_entropies)[-n:]
