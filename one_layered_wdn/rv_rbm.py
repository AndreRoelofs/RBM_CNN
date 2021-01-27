import math

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import transforms


class RV_RBM(nn.Module):
    lowest_energy = 10000
    highest_energy = -10000
    energy_threshold = None
    def __init__(self,
                 n_vis,
                 n_hin,
                 weight_variance,
                 k=1):
        super(RV_RBM, self).__init__()
        self.W = nn.Parameter(torch.zeros(n_hin, n_vis).cuda())
        self.v_bias = nn.Parameter(torch.zeros(n_vis).cuda())
        self.h_bias = nn.Parameter(torch.zeros(n_hin).cuda())
        self.k = k
        #
        nn.init.xavier_normal_(self.W, weight_variance)

    def is_familiar(self, v0, provide_value=True):
        if self.energy_threshold is None:
            return 0
        energy = self.free_energy(v0)

        if torch.isinf(energy).any():
            print("Infinite energy")
            exit(1)

        if provide_value:
            # return torch.where(self.energy_threshold >= energy, 1, 0)
            return self.energy_threshold - energy, torch.where(self.energy_threshold >= energy, 1, 0)
        else:
            return torch.sum(torch.where(self.energy_threshold >= energy, 1, 0))

    def sample_from_p(self, p):
        return F.relu(torch.sign(p - Variable(torch.rand(p.size()).cuda())))

    def v_to_h(self, v):
        p_h = (F.linear(v, self.W, self.h_bias))
        # p_h = torch.sigmoid(p_h)
        sample_h = self.sample_from_p(p_h)
        return p_h, sample_h

    def h_to_v(self, h):
        p_v = (F.linear(h, self.W.t(), self.v_bias))
        # p_v = torch.sigmoid(p_v)
        sample_v = self.sample_from_p(p_v)
        return p_v, sample_v

    def forward(self, v):
        pre_h1, h1 = self.v_to_h(v)

        h_ = h1
        for _ in range(self.k):
            pre_v_, v_ = self.h_to_v(h_)
            pre_h_, h_ = self.v_to_h(v_)
        return v_

    def free_energy(self, v):
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v, self.W, self.h_bias)
        hidden_term = torch.clamp(wx_b, -88, 88).exp().add(1).log().sum(1)
        # hidden_term = wx_b.exp().add(1).log().sum(1)
        return -hidden_term - vbias_term

    def calculate_energy_threshold(self, v0):
        energy = self.free_energy(v0)
        self.lowest_energy = energy.min()
        self.highest_energy = energy.max()
        self.energy_threshold = (self.highest_energy + self.lowest_energy) / 2


