import torch
from torch import nn
import numpy as np
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, channels, filters, weight_mean, weight_variance, use_relu=False):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, filters, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=7, stride=1, padding=3)
        self.conv3 = nn.Conv2d(filters, filters, kernel_size=7, stride=1, padding=3)

        # self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        # self.conv4 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        # self.conv5 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        #
        # nn.init.xavier_normal_(self.conv1.weight, 0.0001)
        nn.init.normal_(self.conv1.weight, weight_mean, weight_variance)
        nn.init.normal_(self.conv2.weight, weight_mean, weight_variance)
        nn.init.normal_(self.conv3.weight, weight_mean, weight_variance)
        # nn.init.normal_(self.conv4.weight, weight_mean, weight_variance)
        # nn.init.normal_(self.conv5.weight, weight_mean, weight_variance)

        #
        self.conv1.weight.requires_grad = False
        self.conv1.weight /= self.conv1.weight.sum()
        self.conv1.weight.requires_grad = True

        self.conv2.weight.requires_grad = False
        self.conv2.weight /= self.conv2.weight.sum()
        self.conv2.weight.requires_grad = True

        self.conv3.weight.requires_grad = False
        self.conv3.weight /= self.conv3.weight.sum()
        self.conv3.weight.requires_grad = True

        if use_relu:
            self.act = nn.ReLU()
        else:
            self.act = nn.SELU()

        # self.pool = nn.MaxPool2d(2)

        # for p in self.parameters():
        #     p.data.clamp_(0)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        # x = self.act(self.conv4(x))
        # x = self.act(self.conv5(x))
        # return torch.sigmoid(x)
        return x

    def loss_function(self, x, recon_x):
        return F.mse_loss(x, recon_x)

