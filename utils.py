import torch
import numpy as np
import random

class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        a = len(self.losses)
        b = np.maximum(a-self.num, 0)
        c = self.losses[b:]

        return torch.mean(torch.stack(c))


def set_seed(option):
    seed = option['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.set_device(option['device'])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True

