# %% Imports

import torch
import torchvision
from torch import nn
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.transforms.functional import resize
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import os
import sys
from torch.utils.data.sampler import SubsetRandomSampler
from siren_pytorch import Sine
from rbm_example.rbm import RBM
np.set_printoptions(threshold=sys.maxsize)


# %%
batch_size = 100
epochs = 1
rbm_epochs = 1
ae_epochs = 0
rbm_epochs_single = 1
target_digit = 1
# RBM_VISIBLE_UNITS = 128 * 7 * 7
# RBM_VISIBLE_UNITS = 64 * 14 * 14
filters = 16
# RBM_VISIBLE_UNITS = filters * 14**2
size = 14
RBM_VISIBLE_UNITS = filters * size**2
# RBM_VISIBLE_UNITS = 1 * 28 * 28
variance = 0.07
RBM_HIDDEN_UNITS = 200
torch.manual_seed(0)
np.random.seed(0)

# %% Load data
train_data = MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()]))

subset_indices = (train_data.targets == target_digit).nonzero().view(-1)

# mask = train_data.targets == target_digit
# indices = torch.nonzero(mask)
#
# train_data.data = train_data.data[indices]
# train_data.targets = train_data.targets[indices]

# mask = target == target_digit
# indices = torch.nonzero(target[mask])
# target = target[indices]

test_data = MNIST('./data', train=False, transform=transforms.Compose([
    transforms.ToTensor()]))


# %% Define encoder
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, filters, (3, 3), stride=1, padding=1)
        # self.conv2 = nn.Conv2d(16, 32, (3, 3), stride=1, padding=1)
        # self.conv3 = nn.Conv2d(32, 64, (3, 3), stride=1, padding=1)

        # self.conv1 = nn.Conv2d(1, 4, (3, 3), stride=1, padding=1)
        # self.conv2 = nn.Conv2d(4, 8, (3, 3), stride=1, padding=1)
        # self.conv3 = nn.Conv2d(8, 12, (3, 3), stride=1, padding=1)
        #
        # self.conv1 = nn.Conv2d(1, 3, (3, 3), stride=1, padding=1)
        # self.conv2 = nn.Conv2d(3, 5, (3, 3), stride=1, padding=1)
        # self.conv3 = nn.Conv2d(5, 6, (3, 3), stride=1, padding=1)

        # nn.init.normal_(self.conv1.weight, 0, variance)
        # nn.init.normal_(self.conv2.weight, 0, variance)
        # nn.init.normal_(self.conv3.weight, 0, variance)

        nn.init.xavier_normal_(self.conv1.weight, 2.0)
        # nn.init.xavier_normal_(self.conv2.weight, 0.5)
        # nn.init.xavier_normal_(self.conv3.weight, 0.1)

        self.maxpool = nn.MaxPool2d((2, 2))

        self.rbm = RBM(RBM_VISIBLE_UNITS, RBM_HIDDEN_UNITS,
                       k=1,
                       learning_rate=1e-2,
                       momentum_coefficient=0.0,
                       weight_decay=0.0,
                       use_cuda=True)

        self.act = nn.SELU()
        # self.act = nn.Sigmoid()
        # self.act = nn.ReLU()


    def forward(self, x):
        return self._forward_3(x)

    def _forward_0(self, x):
        return x

    def _forward_1(self, x):
        x = self.act(self.conv1(x))
        return x

    def _forward_2(self, x):
        x = self.act(self.conv1(x))
        x = self.maxpool(x)
        x = self.act(self.conv2(x))
        return x

    def _forward_3(self, x):
        x = self.act(self.conv1(x))
        # x = self.maxpool(x)
        # x = self.act(self.conv2(x))
        # x = self.act(self.conv3(x))

        return x

    def train_rbm(self, x):
        x = resize(x, [size, size])
        flat_x = x.view(len(x), RBM_VISIBLE_UNITS)

        error = 0

        for i in range(rbm_epochs_single):
            error += self.rbm.contrastive_divergence(flat_x)
        return error

    def get_rbm(self, x):
        x = resize(x, [size, size])
        flat_x = x.view(len(x), RBM_VISIBLE_UNITS)
        return self.rbm.contrastive_divergence(flat_x, update_weights=False)



# %% Define Decoder
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # self.conv1 = nn.ConvTranspose2d(6, 5, (3, 3), stride=1, padding=1)
        # self.conv2 = nn.ConvTranspose2d(5, 3, (3, 3), stride=1, padding=1)
        # self.conv3 = nn.ConvTranspose2d(3, 1, (3, 3), stride=1, padding=1)
        #
        # self.conv1 = nn.ConvTranspose2d(12, 8, (3, 3), stride=1, padding=1)
        # self.conv2 = nn.ConvTranspose2d(8, 4, (3, 3), stride=1, padding=1)
        # self.conv3 = nn.ConvTranspose2d(4, 1, (3, 3), stride=1, padding=1)

        # self.conv1 = nn.ConvTranspose2d(64, 32, (3, 3), stride=1, padding=1)
        # self.conv2 = nn.ConvTranspose2d(32, 16, (3, 3), stride=1, padding=1)
        self.conv3 = nn.ConvTranspose2d(filters, 1, (3, 3), stride=1, padding=1)

        # nn.init.zeros_(self.conv1.weight)
        # nn.init.normal_(self.conv1.weight, 0, variance)
        # nn.init.normal_(self.conv2.weight, 0, variance)
        nn.init.normal_(self.conv3.weight, 0, variance)

        self.upsample = nn.Upsample(scale_factor=(2, 2))
        self.act = nn.SELU()
        # self.act = nn.ReLU()

    def forward(self, z):
        # z = self.act(self.conv1(z))
        # z = self.upsample(z)
        # z = self.act(self.conv2(z))
        # z = self.upsample(z)
        z = torch.sigmoid(self.conv3(z))

        return z

def run_test():
    to_output = []

    for data, target in model.test_loader:
        data = data.to(model.device)
        rbm_input = model.model.encode(data)
        # rbm_input = model.model.encoder._forward_2(data)
        # rbm_input = model.model.encoder._forward_1(data)
        # rbm_input = model.model.encoder._forward_0(data).to(model.device)
        output_energies = torch.sum(model.model.encoder.get_rbm(rbm_input), dim=1).cpu().detach()
        target = target.cpu().detach()
        data = data.cpu().detach()

        for i in range(data.shape[0]):
            t = target[i]
            energy = output_energies[i]

            to_output.append([t.numpy(), torch.mean(energy).numpy(), data[i].numpy()])

    to_output = np.array(to_output, dtype=object)
    to_output = to_output[to_output[:, 1].argsort()]

    markers = ['o', '.', 'x', '+', 'v', '^', '<', '>', 's', 'd']

    for i in [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9

    ]:
        # x = [(j + 1) ** 0.1 for j, e in enumerate(to_output) if int(e[0]) == i]
        x = [(j + 1) ** 0.1 for j, e in enumerate(to_output[:1000]) if int(e[0]) == i]
        plt.plot(x, np.random.uniform(-20, 20, len(x)), markers[i], label="{}".format(i))
    plt.legend(numpoints=1)
    plt.show()

    target_digit_indices = [i for i, e in enumerate(to_output) if int(e[0]) == target_digit]

    # print(target_digit_indices)

    print("{} test: {}".format(len(target_digit_indices), target_digit_indices[-1]-len(target_digit_indices)))
    print("500 test: {}".format(target_digit_indices[500]-500))
    print("100 test: {}".format(target_digit_indices[100]-100))

def run_max_test():
    to_output = []

    for data, target in model.test_loader:
        data = data.to(model.device)
        rbm_input = model.model.encode(data)
        # rbm_input = model.model.encoder._forward_2(data)
        # rbm_input = model.model.encoder._forward_1(data)
        # rbm_input = model.model.encoder._forward_0(data).to(model.device)
        output_energies = model.model.encoder.get_rbm(rbm_input).cpu().detach()
        target = target.cpu().detach()
        data = data.cpu().detach()

        for i in range(data.shape[0]):
            t = target[i]
            energy = output_energies[i]

            to_output.append([t.numpy(), energy.max().numpy(), data[i].numpy()])

    to_output = np.array(to_output, dtype=object)
    to_output = to_output[to_output[:, 1].argsort()]



    # print(to_output[:, 0])

    target_digit_indices = [i for i, e in enumerate(to_output) if int(e[0]) == target_digit]

    # print(target_digit_indices)

    print("500 max test: {}".format(target_digit_indices[500]-500))
    print("500 max test: {}".format(target_digit_indices[500]-500))
    print("100 max test: {}".format(target_digit_indices[100]-100))



# %% Create Autoencoder

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        # self.decoder = Decoder()

    def encode(self, x):
        return self.encoder(x)

    # def decode(self, z):
    #     return self.decoder(z)

    def forward(self, x):
        return self.encode(x)


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda")
        # self.device = torch.device("cpu")
        self.model = Network()
        self.model.to(self.device)

        self.log_interval = 100

        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False,
                                                        sampler=SubsetRandomSampler(subset_indices))
        # self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)

        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def loss_function(self, recon_x, x):
        return F.mse_loss(x, recon_x)
        # return F.binary_cross_entropy(recon_x, x, reduction='sum')

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            rbm_input = self.model(data)
            rbm_input_x = resize(rbm_input, [size, size])
            flat_rbm_input = rbm_input_x.view(len(rbm_input_x), RBM_VISIBLE_UNITS)
            hidden = self.model.encoder.rbm.sample_hidden(flat_rbm_input)
            visible = self.model.encoder.rbm.sample_visible(hidden).reshape((data.shape[0], filters, size, size))
            loss = self.loss_function(visible, rbm_input_x)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader),
                           loss.item() / len(data)))

        print('====> AE Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(self.train_loader.dataset)))

    def train_rbm(self):
        self.model.eval()
        with torch.no_grad():
            train_loss = 0
            for batch_idx, (data, _) in enumerate(self.train_loader):
                data = data.to(self.device)
                rbm_input = self.model.encode(data)
                # rbm_input = self.model.encoder._forward_2(data)
                # rbm_input = self.model.encoder._forward_1(data)
                # rbm_input = self.model.encoder._forward_0(data)
                train_loss += torch.mean(self.model.encoder.train_rbm(rbm_input))
            print('====> RBM Epoch: {} Average loss: {:.4f}'.format(
                epoch, train_loss / len(self.train_loader.dataset)))

    def test(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(self.test_loader):
                data = data.to(self.device)
                recon_batch = self.model(data)
                test_loss += self.loss_function(recon_batch, data).item()

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))


# %% Instantiate the model

model = AE()
run_test()
for epoch in range(epochs):
    for rbm_epoch in range(rbm_epochs):
        model.train_rbm()
    for ae_epoch in range(ae_epochs):
        model.train(epoch)
    run_test()
    # run_max_test()


exit(0)
# model.test()

# %% Visualise data
num_images = 10

num_row = 2
num_col = num_images

images = []
labels = []
energies = []

for data, target in model.test_loader:
    used_images = data[:num_images, :, :, :]
    used_images = used_images.to(model.device)
    output = model.model(used_images)
    output_images = output
    rbm_input = model.model.encode(used_images)
    # rbm_input = model.model.encoder._forward_2(used_images)
    # rbm_input = model.model.encoder._forward_1(used_images)
    # rbm_input = model.model.encoder._forward_0(data).to(model.device)
    output_energies = model.model.encoder.get_rbm(rbm_input)

    # for i in range(used_images.shape[0]):
    #     image = used_images[i]
    #     images.append(image[0].cpu().detach().numpy())
    #     energy = torch.sum(output_energies[i])
    #     labels.append(0)

    for i in range(used_images.shape[0]):
        label = output_images[i]
        images.append(label[0].cpu().detach().numpy())
        energy = torch.mean(output_energies[i])
        labels.append(target[i].detach().numpy())
        energies.append(np.around(energy.cpu().detach().numpy(), 5))

    if num_images * 2 <= len(images):
        images = images[:num_images*2]
        labels = labels[:num_images*2]
        energies = energies[:num_images*2]
        break

fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
for i in range(num_images * 2):
    ax = axes[i // num_col, i % num_col]
    ax.imshow(images[i], cmap='gray')
    ax.set_title('L: {}, E: {}'.format(str(labels[i]), str(energies[i])))
plt.tight_layout()
plt.show()
#%% Print out experiment results



run_test()





