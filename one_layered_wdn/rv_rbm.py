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
                 num_visible,
                 num_hidden,
                 weight_mean,
                 weight_variance,
                 k=1,
                 learning_rate=1e-3,
                 momentum_coefficient=0.9,
                 weight_decay=1e-4,
                 # weight_decay=0,
                 use_relu=True,
                 use_cuda=True
                 ):
        super(RV_RBM, self).__init__()

        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay
        self.use_relu = use_relu
        self.use_cuda = use_cuda

        self.weights = torch.zeros(self.num_visible, self.num_hidden)
        # self.visible_bias = torch.ones(self.num_visible) * 0.5
        self.visible_bias = torch.zeros(self.num_visible)
        self.hidden_bias = torch.zeros(self.num_hidden)

        self.weights_momentum = torch.zeros(self.num_visible, self.num_hidden)
        self.visible_bias_momentum = torch.zeros(self.num_visible)
        self.hidden_bias_momentum = torch.zeros(self.num_hidden)

        # nn.init.xavier_normal_(self.weights, weight_variance)
        nn.init.normal_(self.weights, weight_mean, weight_variance)

        if self.use_cuda:
            self.weights = self.weights.cuda()
            self.visible_bias = self.visible_bias.cuda()
            self.hidden_bias = self.hidden_bias.cuda()

            self.weights_momentum = self.weights_momentum.cuda()
            self.visible_bias_momentum = self.visible_bias_momentum.cuda()
            self.hidden_bias_momentum = self.hidden_bias_momentum.cuda()

        if use_relu:
            self.act = nn.ReLU()
            self.act_prob = torch.sigmoid
            self.random_prob = self._random_relu_probabilities
        else:
            self.act = nn.SELU()
            self.act_prob = torch.tanh
            self.random_prob = self._random_selu_probabilities

    def __call__(self, x):
        positive_hidden_probabilities = self.sample_hidden(x)
        positive_hidden_activations = (
                positive_hidden_probabilities >= self.random_prob(self.num_hidden)).float()
        hidden_activations = positive_hidden_activations
        visible_probabilities = self.sample_visible(hidden_activations)
        return visible_probabilities

    def sample_hidden(self, visible_probabilities):
        hidden_activations = torch.mm(visible_probabilities, self.weights) + self.hidden_bias
        hidden_probabilities = self.act_prob(hidden_activations)
        return self.act(hidden_probabilities)

    def sample_visible(self, hidden_probabilities):
        visible_activations = torch.mm(hidden_probabilities, self.weights.t()) + self.visible_bias
        visible_probabilities = self.act_prob(visible_activations)
        return self.act(visible_probabilities)

    def contrastive_divergence(self, input_data):
        # Positive phase
        positive_hidden_probabilities = self.sample_hidden(input_data)
        positive_hidden_activations = (
                positive_hidden_probabilities >= self.random_prob(self.num_hidden)).float()
        positive_associations = torch.mm(input_data.t(), positive_hidden_activations)

        # Negative phase
        hidden_activations = positive_hidden_activations

        for step in range(self.k):
            visible_probabilities = self.sample_visible(hidden_activations)
            hidden_probabilities = self.sample_hidden(visible_probabilities)
            hidden_activations = (hidden_probabilities >= self.random_prob(self.num_hidden)).float()

        negative_visible_probabilities = visible_probabilities
        negative_hidden_probabilities = hidden_probabilities

        negative_associations = torch.mm(negative_visible_probabilities.t(), negative_hidden_probabilities)

        # Update parameters
        self.weights_momentum *= self.momentum_coefficient
        self.weights_momentum += (positive_associations - negative_associations)

        self.visible_bias_momentum *= self.momentum_coefficient
        self.visible_bias_momentum += torch.sum(input_data - negative_visible_probabilities, dim=0)

        self.hidden_bias_momentum *= self.momentum_coefficient
        self.hidden_bias_momentum += torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0)

        batch_size = input_data.size(0)

        self.weights += self.weights_momentum * self.learning_rate / batch_size
        self.visible_bias += self.visible_bias_momentum * self.learning_rate / batch_size
        self.hidden_bias += self.hidden_bias_momentum * self.learning_rate / batch_size

        self.weights -= self.weights * self.weight_decay  # L2 weight decay

        # Compute reconstruction error
        error = torch.sum((input_data - negative_visible_probabilities) ** 2)

        return error

    def _random_relu_probabilities(self, num):
        random_probabilities = torch.rand(num)

        if self.use_cuda:
            random_probabilities = random_probabilities.cuda()

        return random_probabilities

    def _random_selu_probabilities(self, num):
        random_probabilities = -2 * torch.rand(num) + 1

        if self.use_cuda:
            random_probabilities = random_probabilities.cuda()

        return random_probabilities

    def free_energy(self, v):
        vbias_term = v.mv(self.visible_bias)
        wx_b = F.linear(v, self.weights.t(), self.hidden_bias)
        hidden_term = torch.clamp(wx_b, -88, 88).exp().add(1).log().sum(1)
        # hidden_term = wx_b.exp().add(1).log().sum(1)
        return -hidden_term - vbias_term

    def calculate_energy_threshold(self, v0):
        energy = self.free_energy(v0)
        self.lowest_energy = energy.min()
        self.highest_energy = energy.max()
        self.energy_threshold = (self.highest_energy + self.lowest_energy) / 2

    def is_familiar(self, v0, provide_value=True):
        if self.energy_threshold is None:
            return 0
        energy = self.free_energy(v0)

        if torch.isinf(energy).any():
            print("Infinite energy")
            exit(1)

        if provide_value:
            return self.energy_threshold - energy, torch.where(self.energy_threshold >= energy, 1, 0)
        else:
            return torch.sum(torch.where(self.energy_threshold >= energy, 1, 0))
