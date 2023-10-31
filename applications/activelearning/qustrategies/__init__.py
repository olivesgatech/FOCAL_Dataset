import numpy as np
from applications.activelearning.qustrategies.framestrat import *
from applications.activelearning.qustrategies.seqstrat import *
from config import BaseConfig


def get_frame_sampler(cfg: BaseConfig, n_pool: int, start_idxs: np.ndarray):
    if cfg.active_learning.strategy == 'random':
        print('Using Random Sampler')
        sampler = FrameRandomSampling(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'entropy':
        print('Using Entropy Sampler')
        sampler = FrameEntropySampling(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'lconf':
        print('Using Least Confidence Sampler')
        sampler = FrameLConfSampling(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'margin':
        print('Using Least Margin Sampler')
        sampler = FrameMarginSampling(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    else:
        raise Exception("Frame Sampler not implemented yet")

    return sampler


def get_sequence_sampler(cfg: BaseConfig, n_pool: int, start_idxs: np.ndarray, event_dict: dict = None,
                         addons: dict = None, meta_dict=None, event_list=None, minimum=None):
    if cfg.active_learning.strategy == 'random':
        print('Using Random Sampler')
        sampler = SequenceRandomSampling(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg, event_dict=event_dict)
    elif cfg.active_learning.strategy == 'entropy':
        print('Using Entropy Sampler')
        sampler = SequenceEntropySampling(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg, event_dict=event_dict,
                                          addons=addons)
    elif cfg.active_learning.strategy == 'margin':
        print('Using Margin Sampler')
        sampler = SequenceEntropySampling(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg, event_dict=event_dict,
                                          addons=addons)
    elif cfg.active_learning.strategy == 'lconf':
        print('Using Least Conf. Sampler')
        sampler = SequenceEntropySampling(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg, event_dict=event_dict,
                                          addons=addons)
    elif cfg.active_learning.strategy == 'gauss':
        print('Using GauSS Sampler')
        sampler = SequenceGauss(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg, event_dict=event_dict, addons=addons)
    elif cfg.active_learning.strategy == 'leastframe':
        print('Using Least Frame Sampler')
        sampler = LeastFrameSampling(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg, event_dict=event_dict)
    elif cfg.active_learning.strategy == 'mostframe':
        print('Using Most Frame Sampler')
        sampler = MostFrameSampling(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg, event_dict=event_dict)
    elif cfg.active_learning.strategy == 'minmaxmotion':
        print('Using Min Max Motion Sampler')
        sampler = MinMaxMotionSampling(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg, event_dict=event_dict,
                                       meta_dict=meta_dict, event_list=event_list, minimum=minimum)
    elif cfg.active_learning.strategy == 'minmotion':
        print('Using Min Motion Sampler')
        sampler = MinMotionSampling(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg, event_dict=event_dict,
                                       meta_dict=meta_dict, event_list=event_list)
    elif cfg.active_learning.strategy == 'minboxes':
        print('Using Min Box Estimate Sampler')
        sampler = MinBoxSampling(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg, event_dict=event_dict,
                                       meta_dict=meta_dict, event_list=event_list)
    else:
        raise Exception("Sequence Sampler not implemented yet")

    return sampler
