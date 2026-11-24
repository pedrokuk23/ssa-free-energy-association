import numpy as np
def roc(pos, neg, lower_is_pos=True):
    s = np.concatenate([pos, neg]); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    o = np.argsort(s if lower_is_pos else -s); y = y[o]
    return np.r_[0, np.cumsum(1 - y) / len(neg)], np.r_[0, np.cumsum(y) / len(pos)]


