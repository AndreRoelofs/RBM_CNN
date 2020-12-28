import torch
from torch import nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, channels, filters, use_relu=False):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, filters, (3, 3), stride=1, padding=1)

        nn.init.xavier_normal_(self.conv1.weight, 20.0)
        # nn.init.xavier_normal_(self.conv1.weight, 0.07)

        if use_relu:
            self.act = nn.ReLU()
        else:
            self.act = nn.SELU()

        # for p in self.parameters():
        #     p.data.clamp_(0)

    def forward(self, x):
        x = self.act(self.conv1(x))
        return x
