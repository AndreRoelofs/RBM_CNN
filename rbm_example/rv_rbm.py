import torch
from torch import nn
import numpy as np

counter = 0
total_counter = 0
first_energy = None

# Real valued RBM using Rectified Linear Units
class RV_RBM():
    is_set = False

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

        self.weights = torch.ones((self.num_visible, self.num_hidden), dtype=torch.float)
        # self.weights = torch.randn(num_visible, num_hidden) * 0.1
        # nn.init.xavier_normal_(self.weights, 2.0)
        nn.init.normal_(self.weights, 0, 0.007)

        self.visible_bias = torch.ones(num_visible)
        # self.visible_bias = torch.zeros(num_visible)
        self.hidden_bias = torch.zeros(num_hidden)

        self.weights_momentum = torch.zeros(num_visible, num_hidden)
        self.visible_bias_momentum = torch.zeros(num_visible)
        self.hidden_bias_momentum = torch.zeros(num_hidden)

        if self.use_cuda:
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
        visible_activations = self.act(torch.sign(visible_probabilities - self.rand(visible_probabilities.shape).cuda()))
        return visible_activations

    def contrastive_divergence(self, v0, update_weights=True):
        global first_energy
        global counter
        global total_counter
        batch_size = v0.shape[0]

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

            self.weights += self.weights_momentum * self.lr / batch_size
            self.visible_bias += self.visible_bias_momentum * self.lr / batch_size
            self.hidden_bias += self.hidden_bias_momentum * self.lr / batch_size

            self.weights -= self.weights * self.weight_decay  # L2 weight decay

            # CD = (positive_grad - negative_grad) / batch_size
            #
            # self.weights += self.lr * CD
            # self.visible_bias += self.lr * torch.mean(recon_error, dim=0)
            # self.hidden_bias += self.lr * torch.mean(h0 - h1, dim=0)
            # print("Energy: ", torch.mean(self.free_energy(v0)))
            # print("Recon : ", torch.mean(recon_error_sum))

        energy = self.free_energy(v0)
        if self.is_set:
            print("Energy:  ", energy.mean())
            for e in energy:
                if e > first_energy:
                    counter += 1
                total_counter += 1
            print("Counter: ", counter)
            print("Total Counter: ", total_counter)

        else:
            first_energy = energy.mean()
            print("First energy: ", first_energy)

        # self.free_energy(v0)
        test = 0
        return recon_error_sum

    def free_energy(self, input_data):
        np_input_data = input_data.cpu().detach().numpy()
        np_weights = self.weights.cpu().detach().numpy()
        np_hidden_bias = self.hidden_bias.cpu().detach().numpy()
        np_visible_bias = self.visible_bias.cpu().detach().numpy()

        wx_b = np.dot(np_input_data, np_weights) + np_hidden_bias
        vbias_term = np.dot(np_input_data, np_visible_bias)
        hidden_term = np.sum(np.log(1 + np.exp(wx_b)), axis=1)

        # wx_b = torch.mm(input_data, self.weights) + self.hidden_bias
        # vbias_term = torch.mm(input_data, self.visible_bias)
        # hidden_term = torch.sum(torch.log(1 + torch.exp(wx_b)), dim=1)
        return torch.Tensor(-hidden_term - vbias_term)

    def random_selu_noise(self, shape):
        return -(-2 * torch.rand(shape) + 1).cuda()

    def random_relu_noise(self, shape):
        return torch.rand(shape).cuda()

