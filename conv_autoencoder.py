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
from torchvision.transforms.functional import resize
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
rbm_epochs = 1
ae_epochs = 0
use_relu = False
rbm_epochs_single = 1
# target_digit = 0
# RBM_VISIBLE_UNITS = 128 * 7 * 7
# RBM_VISIBLE_UNITS = 64 * 14 * 14
filters = 8
# RBM_VISIBLE_UNITS = filters * 14**2
size = 14
RBM_VISIBLE_UNITS = filters * size ** 2
MAX_THRESHOLD = 50
# RBM_VISIBLE_UNITS = 1 * 28 * 28
variance = 0.07
RBM_HIDDEN_UNITS = 5
torch.manual_seed(0)
np.random.seed(0)

# %% Load data
transform = transforms.Compose(
    [transforms.ToTensor(),
     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     # transforms.Normalize(0.5, 0.5),
     transforms.RandomResizedCrop((14, 14)),
     transforms.RandomAffine(15, translate=(0, 1), scale=(0.7, 1.0)),
     transforms.RandomVerticalFlip(),
     transforms.RandomHorizontalFlip(),
     ])
# train_data = MNIST('./data', train=True, download=True,
#                    transform=transform)
train_data = MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()]))

# subset_indices = (
#     (torch.tensor(train_data.targets) == target_digit)
#     # + (torch.tensor(train_data.targets) == 8)
# ).nonzero().view(-1)
# subset_indices = subset_indices[torch.randperm(subset_indices.size()[0])]


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
        0,
        # 1,
        # 2,
        # 3,
        # 4,
        # 5,
        # 6,
        # 7,
        # 8,
        # 9,
        # target_digit
    ]:
        x = [e[1] for j, e in enumerate(to_output) if int(e[0]) == i]
        # x = [e[1] for j, e in enumerate(to_output) if int(e[0]) == i]
        # x = [(j + 1) ** 0.1 for j, e in enumerate(to_output[:500]) if int(e[0]) == i]
        plt.plot(x, np.random.uniform(-20, 20, len(x)), markers[i], label="{}".format(i))
    plt.legend(numpoints=1)
    plt.show()

    # target_digit_indices = [i for i, e in enumerate(to_output) if int(e[0]) == 6 or int(e[0]) == 8 or int(e[0]) == 9]
    # target_digit_indices = [i for i, e in enumerate(to_output) if int(e[0]) == target_digit]

    # print(target_digit_indices)

    # print("{} test: {}".format(len(target_digit_indices), target_digit_indices[-1] - len(target_digit_indices)))
    # print("500 test: {}".format(target_digit_indices[500] - 500))
    # print("100 test: {}".format(target_digit_indices[100] - 100))


# %% Define encoder
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, filters, (3, 3), stride=1, padding=1)

        # nn.init.normal_(self.conv1.weight, 0, 0.07)
        # nn.init.normal_(self.conv1.weight, 0, 0.0007)
        # nn.init.xavier_normal_(self.conv1.weight, 0.007)
        nn.init.xavier_normal_(self.conv1.weight, 20.0)

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
                          learning_rate=1e-20,
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


# %%
class Classifier(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.device = torch.device("cuda")

        self.fc1 = nn.Linear(n_features, 500)
        self.fc2 = nn.Linear(500, 10)

        self.act = nn.SELU()

        self.to(self.device)

    def forward(self, x):
        x = self.act(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

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

    def joint_training(self):
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
            # if a_n_models >= 40:
            #     break
            #
            # if counter >= 100:
            #     break

            n_familiar = 0
            familiar_threshold = MAX_THRESHOLD
            for m in self.models:
                # Encode the image
                rbm_input = m.encode(data)
                # Resize and flatten input for RBM
                rbm_input = resize(rbm_input, [size, size])
                flat_rbm_input = rbm_input.view(len(rbm_input), RBM_VISIBLE_UNITS)

                # Compare data with existing models
                if m.rbm.is_familiar(flat_rbm_input, provide_value=False):
                    n_familiar += 1
                if n_familiar >= familiar_threshold:
                    break
            if n_familiar >= familiar_threshold:
                continue

            # If data is unfamiliar, create a new network
            network = self.create_new_model()
            self.model = network
            self.model.train()

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-15)

            for i in range(2):
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
    print("Epoch: ", epoch)
    # model.train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    model.train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=False)
    model.joint_training()
    # run_test()

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
        latent_vector.append(m.rbm.is_familiar(flat_rbm_input).cpu().detach().numpy())

    latent_vector = np.array(latent_vector)
    target_labels = target.cpu().detach().numpy()
    for i in range(classifier_training_batch_size):
        test_target = np.zeros(10, dtype=float)
        test_target[target_labels[i]] = 1.0
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

# %% Training classifier
print("Training classifier")
train_dataset = UnsupervisedVectorDataset(training_features_norm, training_labels)
train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=False)

clf = Classifier(training_features_norm.shape[1])
# criterion = nn.NLLLoss()
# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3)
for epoch in range(5):
    for batch_idx, (data, target) in enumerate(train_dataset_loader):
        data = data.to(clf.device)
        target = target.to(clf.device)

        optimizer.zero_grad()
        out = clf(data)

        loss = clf.loss_function(out, target)
        # loss = clf.loss_function(out, target.long())
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_dataset_loader.dataset),
                       100.0 * batch_idx / len(train_dataset_loader), loss.item()))



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
        latent_vector.append(m.rbm.is_familiar(flat_rbm_input).cpu().detach().numpy())
        # if m.rbm.is_familiar(flat_rbm_input):
        #     latent_vector.append(1)
        # else:
        #     latent_vector.append(0)

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
# %%
print("Making predictions")
test_dataset = UnsupervisedVectorDataset(test_features_norm, test_labels)
test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
test_loss = 0
correct = 0
for batch_idx, (data, target) in enumerate(test_dataset_loader):
    data = data.to(clf.device)
    target = target.to(clf.device)

    optimizer.zero_grad()
    out = clf(data)
    # test_loss += clf.loss_function(out, target.long()).item()
    test_loss += clf.loss_function(out, target).item()
    pred = out.data.max(1)[1]
    target_pred = target.data.max(1)[1]
    correct += pred.eq(target_pred).sum()
    # correct += pred.eq(target.data).sum()


test_loss /= len(test_dataset_loader.dataset)
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_dataset_loader.dataset),
        100. * correct / len(test_dataset_loader.dataset)))

# predictions = clf.predict(test_features_norm)

# print('Result: %d/%d' % (sum(predictions == test_labels), test_labels.shape[0]))
