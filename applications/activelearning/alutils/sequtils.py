import numpy as np


def seq_to_idxs(event_dict: dict, idxs: np.array):
    events = np.array(list(event_dict.keys()))[idxs]
    out = []
    for event in events:
        out.extend(event_dict[event])
    return out
