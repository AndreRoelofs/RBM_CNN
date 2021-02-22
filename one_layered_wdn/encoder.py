import torch
from torch import nn
import numpy as np
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, channels, filters, weight_variance, use_relu=False):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(filters, filters, (3, 3), stride=1, padding=1)

        # nn.init.xavier_normal_(self.conv1.weight, weight_variance)
        nn.init.normal_(self.conv1.weight, 0, weight_variance)
        # nn.init.xavier_normal_(self.conv2.weight, weight_variance/2)
        # nn.init.xavier_normal_(self.conv1.weight, 0.07)

        if use_relu:
            self.act = nn.ReLU()
        else:
            self.act = nn.SELU()

        # for p in self.parameters():
        #     p.data.clamp_(0)

    def forward(self, x):
        x = self.act(self.conv1(x))
        # x = self.act(self.conv2(x))
        # return torch.sigmoid(x)
        return x

    def loss_function(self, x, recon_x):
        return F.mse_loss(x, recon_x)

