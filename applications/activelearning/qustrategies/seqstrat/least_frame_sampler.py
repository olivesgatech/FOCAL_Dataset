import numpy as np
from applications.activelearning.qustrategies.sampler import Sampler
from config import BaseConfig
import collections


class LeastFrameSampling(Sampler):
    """
    Class for least frame sampling algorithm. Inherits from sampler.
    """
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig, event_dict: dict):
        super(LeastFrameSampling, self).__init__(n_pool, start_idxs, cfg, event_dict)

    def query(self, n: int, trainer):
        """
        Performs a query of sequences with the least number of frames
        """
        indxs = np.where(self.total_pool == 0)[0]  # sequences in unlabeled pool
        seqs = np.array(list(self._event_dict.keys()))
        new_event_dict = {key: self._event_dict[key] for key in seqs[indxs]}  # remove seqs not in unlabeled pool
        sorted_events = collections.OrderedDict(sorted(new_event_dict.items(), key=lambda x: len(x[1]), reverse=False))
        selected_seqs = np.array(list(sorted_events.keys())[:n])
        inds = np.arange(seqs.shape[0])[np.in1d(seqs, selected_seqs)]
        return list(inds)

