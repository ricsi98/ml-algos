import numpy as np


class SGD:

    def __init__(self, lr=1e-3, weight_decay=0) -> None:
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self, param_grad):
        lr, wd = self.lr, self.weight_decay
        for p, g in param_grad:
            if wd != 0:
                g = g + wd * p
            p[:] = p - lr * g