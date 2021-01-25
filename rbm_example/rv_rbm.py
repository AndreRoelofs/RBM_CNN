import torch
from torch import nn
import numpy as np


# Real valued RBM using Rectified Linear Units
class RV_RBM():
    lowest_energy = 10000
    highest_energy = -10000
    energy_threshold = None

    def __init__(self, num_visible, num_hidden, learning_rate=1e-5, momentum_coefficient=0.5, weight_decay=1e-4,
                 use_cuda=True, use_relu=True):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.lr = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda

        if use_relu:
            self.act = nn.ReLU()
            self.act_prob = nn.Sigmoid()
            self.rand = self.random_relu_noise
        else:
            self.act = nn.SELU()
            self.act_prob = nn.Tanh()
            self.rand = self.random_selu_noise

        self.weights = torch.zeros((self.num_visible, self.num_hidden), dtype=torch.float)
        # self.weights = torch.randn(num_visible, num_hidden) * 0.1
        # nn.init.xavier_normal_(self.weights, 2.0)
        # nn.init.xavier_normal_(self.weights, 25.0)
        # nn.init.xavier_normal_(self.weights, 25.0)
        nn.init.xavier_normal_(self.weights, 0.07)
        # nn.init.normal_(self.weights, 0, 0.07)
        #
        self.visible_bias = torch.zeros(num_visible)
        # self.visible_bias = torch.ones(num_visible)
        # self.hidden_bias = torch.ones(num_hidden)
        self.hidden_bias = torch.zeros(num_hidden)

        self.weights_momentum = torch.zeros(num_visible, num_hidden)
        self.visible_bias_momentum = torch.zeros(num_visible)
        self.hidden_bias_momentum = torch.zeros(num_hidden)

        if self.use_cuda:
            self.device = torch.device("cuda")
            self.weights = self.weights.cuda()
            self.visible_bias = self.visible_bias.cuda()
            self.hidden_bias = self.hidden_bias.cuda()

            self.weights_momentum = self.weights_momentum.cuda()
            self.visible_bias_momentum = self.visible_bias_momentum.cuda()
            self.hidden_bias_momentum = self.hidden_bias_momentum.cuda()

    def sample_hidden(self, visible_activations):
        # Visible layer activation
        hidden_probabilities = self.act_prob(torch.matmul(visible_activations, self.weights) + self.hidden_bias)
        # Gibb's Sampling
        hidden_activations = self.act(torch.sign(hidden_probabilities - self.rand(hidden_probabilities.shape)))
        return hidden_activations

    def sample_visible(self, hidden_activations):
        visible_probabilities = self.act_prob(torch.matmul(hidden_activations, self.weights.t()) + self.visible_bias)
        visible_activations = self.act(
            torch.sign(visible_probabilities - self.rand(visible_probabilities.shape).cuda()))
        return visible_activations

    def is_familiar(self, v0, provide_value=True):
        if self.energy_threshold is None:
            return 0
        energy = self.free_energy(v0)

        if torch.isinf(energy).any():
            print("Infinite energy")
            exit(1)

        # if energy < self.energy_threshold:
        #     return True
        # return False
        #
        # print(self.energy_threshold - energy)
        # return self.energy_threshold - energy
        if provide_value:
            # return self.energy_threshold - energy
            return energy - self.lowest_energy
        else:
            return torch.sum(torch.where(self.energy_threshold >= energy, 1, 0))

    def contrastive_divergence(self, v0, update_weights=True):
        batch_size = v0.shape[0]

        for i in range(1):
            h0 = self.sample_hidden(v0)
            v1 = self.sample_visible(h0)
            h1 = self.act_prob(torch.matmul(v1, self.weights) + self.hidden_bias)

            positive_grad = torch.matmul(v0.t(), h0)
            negative_grad = torch.matmul(v1.t(), h1)
            recon_error = v0 - v1
            recon_error_sum = torch.mean(recon_error ** 2, dim=1)
            if update_weights:
                CD = (positive_grad - negative_grad)

                self.weights_momentum *= self.momentum_coefficient
                self.weights_momentum += CD

                self.visible_bias_momentum *= self.momentum_coefficient
                self.visible_bias_momentum += torch.sum(recon_error, dim=0)

                self.hidden_bias_momentum *= self.momentum_coefficient
                self.hidden_bias_momentum += torch.sum(h0 - h1, dim=0)

                self.weights = self.weights + (self.weights_momentum * self.lr / batch_size)
                self.visible_bias = self.visible_bias + (self.visible_bias_momentum * self.lr / batch_size)
                self.hidden_bias = self.hidden_bias + (self.hidden_bias_momentum * self.lr / batch_size)

                self.weights = self.weights - (self.weights * self.weight_decay)  # L2 weight decay

        return recon_error_sum

    def calculate_energy_threshold(self, v0):
        energy = self.free_energy(v0)
        self.lowest_energy = min(energy.min(), self.lowest_energy)
        self.highest_energy = max(energy.max(), self.highest_energy)
        self.energy_threshold = (self.highest_energy + self.lowest_energy)/2
        # self.energy_threshold = energy_min
        # print("MIN: ", energy_min)
        # print("MAX: ", energy_max)
        # print(self.energy_threshold)

    def free_energy(self, input_data):
        wx_b = torch.mm(input_data, self.weights) + self.hidden_bias
        vbias_term = torch.sum(input_data * self.visible_bias, axis=1)
        hidden_term = torch.sum(torch.log(1 + torch.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term
        # self.test(input_data[0])
        # np_input_data = input_data.cpu().detach().numpy()
        # np_weights = self.weights.cpu().detach().numpy()
        # np_hidden_bias = self.hidden_bias.cpu().detach().numpy()
        # np_visible_bias = self.visible_bias.cpu().detach().numpy()

        #
        # wx_b = np.dot(np_input_data, np_weights) + np_hidden_bias
        # vbias_term = np.dot(np_input_data, np_visible_bias)
        # hidden_term = np.sum(np.log(1 + np.exp(wx_b)), axis=1)
        #
        # # wx_b = torch.mm(input_data, self.weights) + self.hidden_bias
        # # vbias_term = torch.mm(input_data, self.visible_bias)
        # # hidden_term = torch.sum(torch.log(1 + torch.exp(wx_b)), dim=1)
        # return torch.Tensor(-hidden_term - vbias_term)

    def random_selu_noise(self, shape):
        noise = (-2 * torch.rand(shape) + 1).cuda()
        # noise = torch.where(noise < 0.0, torch.tensor(-0.0, dtype=torch.float).cuda(), noise)
        # noise = torch.where(noise >= 0.0, torch.tensor(0.0, dtype=torch.float).cuda(), noise)

        return noise


    def random_relu_noise(self, shape):
        return torch.rand(shape).cuda()
