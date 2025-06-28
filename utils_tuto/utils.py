import torch
import random
import numpy as np


def seed_all(seed=42):
    torch.manual_seed(seed)  # torch
    random.seed(seed)  # python
    np.random.seed(0)  # numpy
    return None


def get_device():
    if torch.backends.mps.is_available():  # recent Mac users
        device = torch.device("mps")
    elif torch.cuda.is_available():  # cuda
        device = torch.device("cuda:0")
    else:  # default
        device = torch.device("cpu")
    # torch.set_default_device('device')
    # NB : comes at an additional performance cost
    # but avoids explicit move of tensors to MPS device https://docs.pytorch.org/docs/stable/generated/torch.set_default_device.html
    return device
