import numpy as np
from applications.activelearning.qustrategies.sampler import Sampler
from config import BaseConfig


class MinBoxSampling(Sampler):
    """
    Class for min box estimate sampling algorithm. Inherits from sampler.
    """
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig, event_dict: dict, meta_dict=None, event_list=None):
        super(MinBoxSampling, self).__init__(n_pool, start_idxs, cfg, event_dict, meta_dict=meta_dict,
                                                event_list=event_list)

    def query(self, n: int, trainer):
        """
        Performs a query of sequences with the least number of estimated bounding boxes
        """
        indxs = np.where(self.total_pool == 0)[0]  # sequences in unlabeled pool
        updated_meta = self._meta_dict[indxs, :]  # remove seqs not in unlabeled pool
        sorted_motion = updated_meta[:, 2].argsort()  # sort by BoxCountEstimate in increasing order
        start_seqs = updated_meta[sorted_motion[:int(n)]]
        inds = []
        for seq in start_seqs:
            inds.append(self._event_list.index(seq[0].split('-')[-1]))
        return inds

