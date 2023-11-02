import numpy as np
from sklearn.mixture import GaussianMixture
from applications.activelearning.qustrategies.sampler import Sampler
from applications.activelearning.trainer import ALDetectionTrainer
from config import BaseConfig


def gausssampling(switching_list: np.ndarray, cfg: BaseConfig, n: int):
    # init gmm and predict samples
    gmm = GaussianMixture(n_components=2).fit(switching_list.reshape(-1, 1))
    pred = gmm.predict(switching_list.reshape(-1, 1))
    probs = gmm.predict_proba(switching_list.reshape(-1, 1))
    # read current samples from list
    if cfg.active_learning.stats.stat_sampling_type == 'SV':
        if gmm.means_[0][0] < gmm.means_[1][0]:
            relevant_gaussian = 1
        else:
            relevant_gaussian = 0
        backup = np.argsort(switching_list)[len(switching_list) - n:]
    elif cfg.active_learning.stats.stat_sampling_type == 'nSV':
        if gmm.means_[0][0] > gmm.means_[1][0]:
            relevant_gaussian = 1
        else:
            relevant_gaussian = 0
        backup = np.argsort(switching_list)[:n]
    else:
        raise NotImplementedError

    # get relevant indices
    rel_inds = (pred == relevant_gaussian).nonzero()[0]
    length = rel_inds.shape[0]

    # get corresponding probs
    probs = probs[:, relevant_gaussian]
    probs = probs[rel_inds]
    norm_probs = probs / np.sum(probs)

    # check if sufficient samples are available
    if length > n:
        # sample all elements from relevant gaussian with probability distribution
        print('Sampling from RV: ' + cfg.active_learning.stats.stat_sampling_type)
        switching_inds = np.random.choice(rel_inds, size=n, replace=False, p=norm_probs)
    else:
        # not enough elements from prediction include other ones
        print('Not enough samples of forgettable RV or gmm not specified -> using highest/lowest values. Type: '
              + cfg.active_learning.stats.stat_sampling_type)
        switching_inds = backup

    return switching_inds


class SequenceGauss(Sampler):
    """
    Class for random sampling algorithm. Inherits from sampler.
    """
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig, event_dict: dict, addons: dict):
        super(SequenceGauss, self).__init__(n_pool, start_idxs, cfg, event_dict, addons)
        self._idx_event_map = ['' for i in range(addons['total_frames'])]
        event_list = list(event_dict.keys())
        self._event_list = np.array(event_list)
        for i in range(len(event_list)):
            for idx in event_dict[event_list[i]]:
                # print(len(event_list[i]))
                self._idx_event_map[idx] += event_list[i]
        self._idx_event_map = np.array(self._idx_event_map)
        self._cur_switches = None
        self._cur_stats = None

    def query(self, n: int, trainer: ALDetectionTrainer):
        """
        Performs entropy query of points
        """
        stats = self._cur_stats
        if stats is None:
            raise Exception('Gauss refresh rate is larger than training epochs each round')
        relevant_events = {}
        relevant_events_np = np.zeros(self._event_list.shape[0])
        for idx in stats['idxs']:
            relevant_events[self._idx_event_map[idx]] = True
            relevant_events_np[self._event_list == self._idx_event_map[idx]] = 1
        switches = np.zeros(self._event_list.shape[0])
        for event in list(relevant_events.keys()):
            # inds = np.searchsorted(stats['idxs'], self._event_dict[event])
            inds = np.array([np.where(stats['idxs'] == i)[0] for i in self._event_dict[event]])

            # switches for the sequence 'event'
            seq_switches = stats['switches'][inds]
            switch = np.mean(seq_switches)
            switches[self._event_list == event] = switch

        # remove sequences that have been queried
        event_inds = np.arange(self._event_list.shape[0])
        switches = switches[relevant_events_np == 1]
        self._cur_switches = switches
        event_inds = event_inds[relevant_events_np == 1]
        out = gausssampling(switches, self._cfg, n)
        return event_inds[out]

    def action(self, trainer, epoch: int):
        if epoch % self._cfg.active_learning.stats.switch_refresh_rate == 0:
            self._cur_stats = trainer.switch_statistics()

    def save_data(self, target_folder: str):
        """
        save switching inidices
        :param target_folder:
        :return:
        """
        if self._cur_switches is not None:
            np.save(target_folder + 'switches.npy', self._cur_switches)
