import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F


class ClassifierV1(nn.Module): #similar to a basic SVM
    def __init__(self):
        super().__init__()
        self.fully_connected = nn.Linear(4, 1)

    def forward(self, x):
        fwd = self.fully_connected(x)
        return fwd
