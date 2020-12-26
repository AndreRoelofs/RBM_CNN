# %% Imports

import torch
import torchvision
from torch import nn
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.transforms.functional import resize, center_crop, gaussian_blur
from torchvision.datasets import MNIST, CIFAR10
import matplotlib.pyplot as plt
import os
import sys
from torch.utils.data.sampler import SubsetRandomSampler
# from rbm_example.rbm_altered import RBM
from rbm_example.rv_rbm import RV_RBM
import rbm_example.custom_activations

np.set_printoptions(threshold=sys.maxsize)

# %%
train_batch_size = 10
test_batch_size = 100
one_shot_classifier = False
if one_shot_classifier:
    train_batch_size = 1
epochs = 1
use_relu = True
filters = 1
channels = 1
size = 14
RBM_VISIBLE_UNITS = filters * size ** 2
MIN_FAMILIARITY_THRESHOLD = 10
variance = 0.07
RBM_HIDDEN_UNITS = 1
torch.manual_seed(0)
np.random.seed(0)


class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# %% Load data
train_data = MNIST('./data', train=True, download=True,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.CenterCrop(size),
                         GaussianNoise(0., 0.25)
                     ]))

# train_data.data = train_data.data[:10000]
# train_data.targets = train_data.targets[:10000]

test_data = MNIST('./data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(size),
    GaussianNoise(0., 0.1)
]))


# %% Define encoder
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, filters, (3, 3), stride=1, padding=1)

        # nn.init.normal_(self.conv1.weight, 5.0, 1.0)
        # nn.init.normal_(self.conv1.weight, 0, 0.0007)
        # nn.init.xavier_normal_(self.conv1.weight, 0.007)
        # nn.init.xavier_normal_(self.conv1.weight, 5.0)
        nn.init.xavier_normal_(self.conv1.weight, 1.0)

        if use_relu:
            self.act = nn.ReLU()
        else:
            self.act = nn.SELU()

        for p in self.parameters():
            p.data.clamp_(0)

    def forward(self, x):
        x = self.act(self.conv1(x))
        return x


# %% Create Autoencoder

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.rbm = RV_RBM(RBM_VISIBLE_UNITS, RBM_HIDDEN_UNITS,
                          learning_rate=1e-3,
                          momentum_coefficient=0.0,
                          weight_decay=0.00,
                          use_cuda=True,
                          use_relu=use_relu)

    def encode(self, x):
        return self.encoder(x)


# %%
class Classifier(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.device = torch.device("cuda")

        self.fc1 = nn.Linear(n_features, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 10)

        self.fc1_bn = nn.BatchNorm1d(400)
        self.fc2_bn = nn.BatchNorm1d(200)
        self.fc3_bn = nn.BatchNorm1d(100)
        self.fc4_bn = nn.BatchNorm1d(10)

        self.act = nn.SELU()
        # self.act = nn.ReLU()

        # if use_relu:
        #     self.act = nn.ReLU()
        # else:
        #     self.act = nn.SELU()

        self.to(self.device)

    def forward(self, x):
        x = self.fc1_bn(self.fc1(x))
        x = self.act(x)
        x = self.fc2_bn(self.fc2(x))
        x = self.act(x)
        x = self.fc3_bn(self.fc3(x))
        x = self.act(x)
        x = self.fc4(x)
        x = self.fc4_bn(x)

        return F.log_softmax(x, dim=1)

    def loss_function(self, x, y):
        # return F.nll_loss(x, y)
        return F.kl_div(x, y)
        # return F.mse_loss(x, y)


# %%

class UnsupervisedVectorDataset(Dataset):
    def __init__(self, features, labels):
        # self.features = torch.from_numpy(features, dtype=torch.long)
        # self.labels = torch.from_numpy(labels, dtype=torch.long)

        self.features = torch.tensor(features, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return [self.features[idx], self.labels[idx]]


# %%

class WDN(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda")
        # self.device = torch.device("cpu")

        self.models = []
        # self.create_new_model()

        self.log_interval = 100

        # self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=False,
        #                                                 sampler=SubsetRandomSampler(subset_indices))

        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

    def create_new_model(self):
        network = Network()
        self.models.append(network)
        network.to(self.device)

        return network

    def loss_function(self, recon_x, x):
        return F.mse_loss(x, recon_x)
        # return F.binary_cross_entropy(recon_x, x, reduction='sum')

    def joint_training(self, MIN_FAMILIARITY_THRESHOLD):
        # torch.autograd.set_detect_anomaly(True)
        counter = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            # Assume we have batch size of 1
            data = data.to(self.device)

            a_n_models = len(self.models)

            counter += 1
            if counter % 25 == 0:
                print("Iteration: ", counter)
                print("n_models", a_n_models)

            n_familiar = 0
            model_counter = 0
            for m in self.models:
                # Encode the image
                rbm_input = m.encode(data)
                # Resize and flatten input for RBM
                rbm_input = resize(rbm_input, [size, size])
                flat_rbm_input = rbm_input.view(len(rbm_input), RBM_VISIBLE_UNITS)

                # Compare data with existing models
                familiarity = m.rbm.is_familiar(flat_rbm_input, provide_value=False)
                if familiarity > train_batch_size / 2:
                    n_familiar += 1
                if n_familiar >= MIN_FAMILIARITY_THRESHOLD or n_familiar + (
                        a_n_models - model_counter) < MIN_FAMILIARITY_THRESHOLD:
                    break
                model_counter += 1
            if n_familiar >= MIN_FAMILIARITY_THRESHOLD:
                # break
                continue

            # If data is unfamiliar, create a new network
            network = self.create_new_model()
            self.model = network
            self.model.train()

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

            for i in range(50):
                # Encode the image
                rbm_input = self.model.encode(data)
                # Resize and flatten input for RBM
                rbm_input = resize(rbm_input, [size, size])
                flat_rbm_input = rbm_input.view(len(rbm_input), RBM_VISIBLE_UNITS)

                familiarity = self.model.rbm.is_familiar(flat_rbm_input, provide_value=False)
                if familiarity == train_batch_size:
                    self.model.rbm.calculate_energy_threshold(flat_rbm_input)
                    break
                # Train RBM
                rbm_error = self.model.rbm.contrastive_divergence(flat_rbm_input, update_weights=True)

                hidden = self.model.rbm.sample_hidden(flat_rbm_input)
                visible = self.model.rbm.sample_visible(hidden).reshape((data.shape[0], filters, size, size))
                loss = self.loss_function(visible, rbm_input)
                loss.backward(retain_graph=True)
                self.model.rbm.calculate_energy_threshold(flat_rbm_input)


# %% Instantiate the model

model = WDN()

# %% Train the model
for i in range(10):
    print("Training digit: ", i)
    subset_indices = (torch.tensor(train_data.targets) == i).nonzero().view(-1)
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        subset_indices = subset_indices[torch.randperm(subset_indices.size()[0])]
        model.train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=False,
                                                         sampler=SubsetRandomSampler(subset_indices))
        model.joint_training(MIN_FAMILIARITY_THRESHOLD)


# %%

def predict_classifier():
    print("Making predictions")
    test_loss = 0
    correct = 0
    clf.eval()
    for batch_idx, (data, target) in enumerate(test_dataset_loader):
        data = data.to(clf.device)
        target = target.to(clf.device)
        out = clf(data)
        test_loss += clf.loss_function(out, target).item()
        pred = out.data.max(1)[1]
        target_pred = target.data.max(1)[1]
        correct += pred.eq(target_pred).sum()
        # correct += pred.eq(target.data).sum()
    test_loss /= len(test_dataset_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_dataset_loader.dataset),
        100. * correct / len(test_dataset_loader.dataset)))


def train_classifier():
    clf.train()
    for epoch in range(10):
        for batch_idx, (data, target) in enumerate(train_dataset_loader):
            data = data.to(clf.device)
            target = target.to(clf.device)
            optimizer.zero_grad()
            out = clf(data)
            loss = clf.loss_function(out, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_dataset_loader.dataset),
                           100.0 * batch_idx / len(train_dataset_loader), loss.item()))
        predict_classifier()


# %% Convert the training set to the unsupervised latent vector
print("Converting images to latent vectors")
classifier_training_batch_size = 1000
train_loader = torch.utils.data.DataLoader(train_data, batch_size=classifier_training_batch_size, shuffle=False)
counter = 0
training_features = []
training_labels = []
new_dataset = []
for batch_idx, (data, target) in enumerate(train_loader):
    data = data.to(model.device)
    latent_vector = []
    for m in model.models:
        # Encode the image
        rbm_input = m.encode(data)
        # Resize and flatten input for RBM
        rbm_input = resize(rbm_input, [size, size])
        flat_rbm_input = rbm_input.view(len(rbm_input), RBM_VISIBLE_UNITS)

        # Compare data with existing models
        values = m.rbm.is_familiar(flat_rbm_input)

        # subset_indices = (is_familiar == 0).nonzero().view(-1)
        # values[subset_indices] = -20

        values = values.cpu().detach().numpy()

        latent_vector.append(values)

    latent_vector = np.array(latent_vector)
    target_labels = target.cpu().detach().numpy()
    for i in range(classifier_training_batch_size):
        test_target = np.zeros(10, dtype=float)
        test_target[target_labels[i]] = 1.0
        # training_features.append(latent_vector[:, i])
        training_features.append(latent_vector[:, i])
        # training_labels.append(target_labels[i])
        training_labels.append(test_target)

    counter += 1
    if counter % 100 == 0:
        print("Training iteration: ", counter)
# new_dataset = np.array(new_dataset, dtype=float)
training_features = np.array(training_features)
training_features_norm = preprocessing.scale(training_features)
training_labels = np.array(training_labels, dtype=float)
#%%
train_dataset = UnsupervisedVectorDataset(training_features_norm, training_labels)
train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)

# %%
print("Converting test images to latent vectors")
test_batch_size = 1000
test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
counter = 0
test_features = []
test_labels = []
for batch_idx, (data, target) in enumerate(test_loader):
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
        values = m.rbm.is_familiar(flat_rbm_input)

        # subset_indices = (is_familiar == 0).nonzero().view(-1)
        # values[subset_indices] = 0

        values = values.cpu().detach().numpy()

        latent_vector.append(values)

    latent_vector = np.array(latent_vector)
    target_labels = target.cpu().detach().numpy()
    for i in range(test_batch_size):
        test_target = np.zeros(10, dtype=float)
        test_target[target_labels[i]] = 1
        test_features.append(latent_vector[:, i])
        # test_labels.append(target_labels[i])
        test_labels.append(test_target)

    # latent_vector.append(t)
    # new_dataset.append(latent_vector)
    counter += 1
    if counter % 100 == 0:
        print("Testing iteration: ", counter)

    # if counter >= 100:
    #     break
# new_dataset = np.array(new_dataset)

test_features = np.array(test_features)
test_features_norm = preprocessing.scale(test_features)
test_labels = np.array(test_labels)
#%%
test_dataset = UnsupervisedVectorDataset(test_features_norm, test_labels)
test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

# %% Training classifier
print("Training classifier")
clf = Classifier(training_features_norm.shape[1])
# criterion = nn.NLLLoss()
# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3, amsgrad=True)

train_classifier()
