import numpy as np


def compute_forgetting(acc_mat, weights=None):
    """
    forgetting = final acc - max acc reached at any step for a given dataset
    """
    max_acc = np.max(acc_mat, axis=0)
    last_acc = acc_mat[-1]
    f = np.average(max_acc-last_acc, weights=weights)
    return f

