import numpy as np
from applications.activelearning.qustrategies.sampler import Sampler
from applications.activelearning.trainer import ALDetectionTrainer
from config import BaseConfig


class SequenceEntropySampling(Sampler):
    """
    Class for random sampling algorithm. Inherits from sampler.
    """
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig, event_dict: dict, addons: dict):
        super(SequenceEntropySampling, self).__init__(n_pool, start_idxs, cfg, event_dict, addons)
        self._idx_event_map = ['' for i in range(addons['total_frames'])]
        event_list = list(event_dict.keys())
        self._event_list = np.array(event_list)
        for i in range(len(event_list)):
            for idx in event_dict[event_list[i]]:
                # print(len(event_list[i]))
                self._idx_event_map[idx] += event_list[i]
        self._idx_event_map = np.array(self._idx_event_map)

    def query(self,model, n: int, trainer: ALDetectionTrainer):
        """
        Performs entropy query of points
        """
        stats = trainer.unlabeled_statistics(model)
        relevant_events = {}
        for idx in stats['idxs']:
            relevant_events[self._idx_event_map[idx]] = True

        # Note: we are considering events that have already been queried as well but assume a entropy of 0
        entropies = np.zeros(self._event_list.shape[0])
        for event in list(relevant_events.keys()):
            # inds = np.searchsorted(stats['idxs'], self._event_dict[event])
            inds = np.array([np.where(stats['idxs'] == i)[0] for i in self._event_dict[event]])
            seq_entropies = stats['entropy'][inds]
            entropy = np.mean(seq_entropies)
            entropies[self._event_list == event] = entropy

        return np.argsort(entropies)[-n:]
