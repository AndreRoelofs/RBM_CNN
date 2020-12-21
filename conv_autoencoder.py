# %% Imports

import torch
import torchvision
from torch import nn
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.transforms.functional import resize
from torchvision.datasets import MNIST, CIFAR10
import matplotlib.pyplot as plt
import os
import sys
from torch.utils.data.sampler import SubsetRandomSampler
# from rbm_example.rbm_altered import RBM
from rbm_example.rv_rbm import RV_RBM

np.set_printoptions(threshold=sys.maxsize)

# %%
train_batch_size = 100
test_batch_size = 100
one_shot_classifier = False
if one_shot_classifier:
    train_batch_size = 1
epochs = 10
rbm_epochs = 1
ae_epochs = 0
use_relu = False
rbm_epochs_single = 1
target_digit = 5
# RBM_VISIBLE_UNITS = 128 * 7 * 7
# RBM_VISIBLE_UNITS = 64 * 14 * 14
filters = 1
# RBM_VISIBLE_UNITS = filters * 14**2
size = 14
RBM_VISIBLE_UNITS = filters * size ** 2
# RBM_VISIBLE_UNITS = 1 * 28 * 28
variance = 0.07
RBM_HIDDEN_UNITS = 5
# torch.manual_seed(0)
# np.random.seed(0)

# %% Load data
train_data = MNIST('./data', train=False, download=True,
                     transform=transforms.Compose([
                         transforms.ToTensor()]))

subset_indices = (
    (torch.tensor(train_data.targets) == target_digit)
    # + (torch.tensor(train_data.targets) == 8)
).nonzero().view(-1)
subset_indices = subset_indices[torch.randperm(subset_indices.size()[0])]


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


# %%
def run_test():
    to_output = []

    for data, target in model.test_loader:
        data = data.to(model.device)
        rbm_input = model.model.encode(data)
        rbm_input_x = resize(rbm_input, [size, size])
        flat_rbm_input = rbm_input_x.view(len(rbm_input_x), RBM_VISIBLE_UNITS)
        output_energies = model.model.rbm.free_energy(flat_rbm_input)
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
        # 0,
        # 1,
        # 2,
        # 3,
        # 4,
        # 5,
        # 6,
        # 7,
        # 8,
        # 9,
        target_digit
    ]:
        x = [e[1] for j, e in enumerate(to_output) if int(e[0]) == i]
        # x = [e[1] for j, e in enumerate(to_output) if int(e[0]) == i]
        # x = [(j + 1) ** 0.1 for j, e in enumerate(to_output[:500]) if int(e[0]) == i]
        plt.plot(x, np.random.uniform(-20, 20, len(x)), markers[i], label="{}".format(i))
    plt.legend(numpoints=1)
    plt.show()

    # target_digit_indices = [i for i, e in enumerate(to_output) if int(e[0]) == 6 or int(e[0]) == 8 or int(e[0]) == 9]
    target_digit_indices = [i for i, e in enumerate(to_output) if int(e[0]) == target_digit]

    # print(target_digit_indices)

    print("{} test: {}".format(len(target_digit_indices), target_digit_indices[-1] - len(target_digit_indices)))
    print("500 test: {}".format(target_digit_indices[500] - 500))
    print("100 test: {}".format(target_digit_indices[100] - 100))


# %% Define encoder
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, filters, (3, 3), stride=1, padding=1)

        # nn.init.normal_(self.conv1.weight, 0, variance)
        nn.init.normal_(self.conv1.weight, 0, 0.0007)
        # nn.init.xavier_normal_(self.conv1.weight, 0.007)
        # nn.init.xavier_normal_(self.conv1.weight, 2.0)

        if use_relu:
            self.act = nn.ReLU()
        else:
            self.act = nn.SELU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        return x


# %% Create Autoencoder

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.rbm = RV_RBM(RBM_VISIBLE_UNITS, RBM_HIDDEN_UNITS,
                          learning_rate=1e-2,
                          momentum_coefficient=0.0,
                          weight_decay=0.00,
                          use_cuda=True,
                          use_relu=use_relu)

    def encode(self, x):
        return self.encoder(x)

    def train_rbm(self, x, update_weights=True):
        x = resize(x, [size, size])
        flat_x = x.view(len(x), RBM_VISIBLE_UNITS)

        error = 0

        for i in range(rbm_epochs_single):
            error += self.rbm.contrastive_divergence(flat_x, update_weights=update_weights)
        return error

    def get_rbm(self, x):
        x = resize(x, [size, size])
        flat_x = x.view(len(x), RBM_VISIBLE_UNITS)
        return self.rbm.contrastive_divergence(flat_x, update_weights=False)


class WDN(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda")
        # self.device = torch.device("cpu")

        self.models = []
        # self.create_new_model()

        self.log_interval = 100

        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=False,
                                                        sampler=SubsetRandomSampler(subset_indices))
        # self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=False)

        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

    def create_new_model(self):
        network = Network()
        self.models.append(network)
        network.to(self.device)

        return network

    def loss_function(self, recon_x, x):
        return F.mse_loss(x, recon_x)
        # return F.binary_cross_entropy(recon_x, x, reduction='sum')

    def joint_training(self):
        # torch.autograd.set_detect_anomaly(True)
        counter = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            # Assume we have batch size of 1
            data = data.to(self.device)

            a_n_models = len(self.models)

            if counter % 100 == 0:
                print("Iteration: ", counter)
                print("n_models", a_n_models)

            # if counter >= 100:
            #     break

            counter += 1
            n_familiar = 0
            familiar_threshold = 1
            for m in self.models:
                # Encode the image
                rbm_input = m.encode(data)
                # Resize and flatten input for RBM
                rbm_input = resize(rbm_input, [size, size])
                flat_rbm_input = rbm_input.view(len(rbm_input), RBM_VISIBLE_UNITS)

                # Compare data with existing models
                if m.rbm.is_familiar(flat_rbm_input) > 0.9:
                    n_familiar += 1

            if n_familiar >= familiar_threshold:
                continue

            # If data is unfamiliar, create a new network
            network = self.create_new_model()
            self.model = network
            self.model.train()

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-15)

            # Encode the image
            rbm_input = self.model.encode(data)
            # Resize and flatten input for RBM
            rbm_input = resize(rbm_input, [size, size])
            flat_rbm_input = rbm_input.view(len(rbm_input), RBM_VISIBLE_UNITS)

            # Train RBM
            rbm_error = self.model.rbm.contrastive_divergence(flat_rbm_input, update_weights=True)

            # Sample RBM
            hidden = self.model.rbm.sample_hidden(flat_rbm_input)
            visible = self.model.rbm.sample_visible(hidden).reshape((data.shape[0], filters, size, size))

            # Train Encoder
            loss = self.loss_function(visible, rbm_input)
            loss.backward()

            self.model.rbm.calculate_energy_threshold(flat_rbm_input)



# %% Instantiate the model

model = WDN()
# run_test()

for epoch in range(epochs):
    model.joint_training()
    # run_test()


# %% Convert the training set to the unsupervised latent vector
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)
counter = 0
new_dataset = []
training_features = []
training_labels = []
for batch_idx, (data, target) in enumerate(train_loader):
    data = data.to(model.device)

    latent_vector = []

    familiar = False
    for m in model.models:
        # Encode the image
        rbm_input = m.encode(data)
        # Resize and flatten input for RBM
        rbm_input = resize(rbm_input, [size, size])
        flat_rbm_input = rbm_input.view(len(rbm_input), RBM_VISIBLE_UNITS)

        # Compare data with existing models
        latent_vector.append(m.rbm.is_familiar(flat_rbm_input))
        # if m.rbm.is_familiar(flat_rbm_input):
        #     latent_vector.append(1)
        # else:
        #     latent_vector.append(0)

    t = target.cpu().detach().numpy()[0]
    training_features.append(latent_vector)
    # training_labels.append(1 if t == target_digit else 0)
    training_labels.append(t)

    # latent_vector.append(t)
    # new_dataset.append(latent_vector)
    counter += 1
    if counter >= 100:
        break
# new_dataset = np.array(new_dataset)

training_features = np.array(training_features)
training_labels = np.array(training_labels)

clf = LogisticRegression()
clf.fit(training_features, training_labels)
predictions = clf.predict(training_features)

print('Result: %d/%d' % (sum(predictions == training_labels), training_labels.shape[0]))


exit(0)

# %% Visualise data
num_images = 10

num_row = 3
num_col = num_images

images = []
labels = []
energies = []

# for data, target in model.train_loader:
for data, target in model.test_loader:
    used_images = data[:num_images, :, :, :]
    used_images = used_images.to(model.device)
    output = model.model.encode(used_images)
    output_images = output
    rbm_input = model.model.encode(used_images)
    rbm_input_x = resize(rbm_input, [size, size])
    flat_rbm_input = rbm_input_x.view(len(rbm_input_x), RBM_VISIBLE_UNITS)
    output_energies = model.model.rbm.free_energy(flat_rbm_input)

    # for i in range(used_images.shape[0]):
    #     image = used_images[i]
    #     images.append(image[0].cpu().detach().numpy())
    #     energy = torch.sum(output_energies[i])
    #     labels.append(0)

    for i in range(used_images.shape[0]):
        label = output_images[i]
        images.append(label[0].cpu().detach().numpy())
        energy = output_energies[i]
        labels.append(target[i].detach().numpy())
        energies.append(np.around(energy.cpu().detach().numpy(), 3))

    if num_images * num_row <= len(images):
        images = images[:num_images * num_row]
        labels = labels[:num_images * num_row]
        energies = energies[:num_images * num_row]
        break

fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
for i in range(num_images * num_row):
    ax = axes[i // num_col, i % num_col]
    ax.imshow(images[i], cmap='gray')
    ax.set_title('L: {}, E: {}'.format(str(labels[i]), str(energies[i])))
plt.tight_layout()
plt.show()
