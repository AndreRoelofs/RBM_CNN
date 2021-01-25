# %% Imports

import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
from torchvision import transforms
from torchvision.transforms.functional import crop
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
import sys
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from rbm_example.rv_rbm import RV_RBM

np.set_printoptions(threshold=sys.maxsize)

# %%
node_train_batch_size = 1
test_batch_size = 100
one_shot_classifier = False
if one_shot_classifier:
    node_train_batch_size = 1
epochs = 1
use_relu = False
filters = 1
rbm_input_size = 3
RBM_VISIBLE_UNITS = filters * rbm_input_size ** 2
MIN_FAMILIARITY_THRESHOLD = 1
variance = 0.07
image_size = 28
RBM_HIDDEN_UNITS = 1
torch.manual_seed(0)
np.random.seed(0)

# %% Load data
train_data = MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ]))

train_data.data = train_data.data[:1000]
train_data.targets = train_data.targets[:1000]

#%%

test_data = MNIST('../data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
]))

test_data.data = test_data.data[:100]
test_data.targets = test_data.targets[:100]

# %% Define encoder
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, filters, (3, 3), stride=1, padding=1)

        nn.init.xavier_normal_(self.conv1.weight, 0.05)

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
                          learning_rate=1e-10,
                          momentum_coefficient=0.0,
                          weight_decay=0.00,
                          use_cuda=True,
                          use_relu=use_relu)

        self.child_networks = []

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
        # self.fc4_bn = nn.BatchNorm1d(2)

        self.act = nn.SELU()
        self.to(self.device)

    def forward(self, x):
        x = self.fc1_bn(self.fc1(x))
        x = self.act(x)
        x = self.fc2_bn(self.fc2(x))
        x = self.act(x)
        x = self.fc3_bn(self.fc3(x))
        x = self.act(x)
        x = self.fc4(x)
        # x = self.fc4_bn(x)

        return F.log_softmax(x, dim=1)

    def loss_function(self, x, y):
        return F.nll_loss(x, y)


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

        self.models = []
        self.log_interval = 100
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

    def create_new_model(self):
        network = Network()
        network.cuda()
        return network

    def loss_function(self, recon_x, x):
        return F.mse_loss(x, recon_x)

    def joint_training(self, MIN_FAMILIARITY_THRESHOLD):
        # torch.autograd.set_detect_anomaly(True)
        counter = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            a_n_models = len(self.models)

            counter += 1
            if counter % 50 == 0:
                print("Iteration: ", counter)
                print("n_models", a_n_models)

                for m in self.models:
                    print(len(m.child_networks))

            data_cropped_center = crop(data, int(image_size / 2) - rbm_input_size, int(image_size / 2) - rbm_input_size,
                                       rbm_input_size, rbm_input_size)
            regions_to_check = [
                crop(data, int(image_size / 2) - rbm_input_size - rbm_input_size,
                     int(image_size / 2) - rbm_input_size - rbm_input_size, rbm_input_size, rbm_input_size),
                crop(data, int(image_size / 2) - rbm_input_size + rbm_input_size,
                     int(image_size / 2) - rbm_input_size + rbm_input_size, rbm_input_size, rbm_input_size),
                crop(data, int(image_size / 2) - rbm_input_size - rbm_input_size,
                     int(image_size / 2) - rbm_input_size + rbm_input_size, rbm_input_size, rbm_input_size),
                crop(data, int(image_size / 2) - rbm_input_size + rbm_input_size,
                     int(image_size / 2) - rbm_input_size - rbm_input_size, rbm_input_size, rbm_input_size),
            ]

            # go over first level nodes to check centers
            n_familiar = 0
            for m in self.models:
                # Encode the image
                rbm_input = m.encode(data_cropped_center)
                # flatten input for RBM
                flat_rbm_input = rbm_input.view(len(rbm_input), RBM_VISIBLE_UNITS)

                # Compare data with existing models
                familiarity = m.rbm.is_familiar(flat_rbm_input, provide_value=False)
                if familiarity >= 1:
                    for region in regions_to_check:
                        n_child_familiarity = 0
                        for child_m in m.child_networks:
                            # Encode the image
                            rbm_input = child_m.encode(region)
                            # flatten input for RBM
                            flat_rbm_input = rbm_input.view(len(rbm_input), RBM_VISIBLE_UNITS)

                            children_familiarity = m.rbm.is_familiar(flat_rbm_input, provide_value=False)
                            n_child_familiarity += children_familiarity
                            if children_familiarity >= 1:
                                n_familiar += 1
                            if n_familiar >= MIN_FAMILIARITY_THRESHOLD:
                                break
                        if n_child_familiarity >= 1:
                            n_familiar += 1
                            continue
                        network = self.create_new_model()
                        m.child_networks.append(network)
                        self.model = network
                        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-10)
                        self.model.train()
                        for i in range(5):
                            rbm_input = self.model.encode(region)
                            # Flatten input for RBM
                            flat_rbm_input = rbm_input.view(len(rbm_input), RBM_VISIBLE_UNITS)

                            familiarity = self.model.rbm.is_familiar(flat_rbm_input, provide_value=False)
                            if familiarity == node_train_batch_size:
                                self.model.rbm.calculate_energy_threshold(flat_rbm_input)
                                break
                            # Train RBM
                            rbm_error = self.model.rbm.contrastive_divergence(flat_rbm_input, update_weights=True)
                            hidden = self.model.rbm.sample_hidden(flat_rbm_input)
                            visible = self.model.rbm.sample_visible(hidden).reshape(
                                (data.shape[0], filters, rbm_input_size, rbm_input_size))
                            loss = self.loss_function(visible, rbm_input)
                            loss.backward(retain_graph=True)
                            self.model.rbm.calculate_energy_threshold(flat_rbm_input)
                        if n_familiar >= MIN_FAMILIARITY_THRESHOLD:
                            break
            if n_familiar >= MIN_FAMILIARITY_THRESHOLD:
                # break
                continue

            # If data is unfamiliar, create a new network
            network = self.create_new_model()
            self.models.append(network)
            self.model = network
            self.model.train()

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-10)

            for i in range(5):
                # Encode the image
                rbm_input = self.model.encode(data_cropped_center)
                # Flatten input for RBM
                flat_rbm_input = rbm_input.view(len(rbm_input), RBM_VISIBLE_UNITS)

                familiarity = self.model.rbm.is_familiar(flat_rbm_input, provide_value=False)
                if familiarity == node_train_batch_size:
                    self.model.rbm.calculate_energy_threshold(flat_rbm_input)
                    break
                # Train RBM
                rbm_error = self.model.rbm.contrastive_divergence(flat_rbm_input, update_weights=True)

                hidden = self.model.rbm.sample_hidden(flat_rbm_input)
                visible = self.model.rbm.sample_visible(hidden).reshape(
                    (data.shape[0], filters, rbm_input_size, rbm_input_size))
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
        model.train_loader = torch.utils.data.DataLoader(train_data, batch_size=node_train_batch_size, shuffle=False,
                                                         sampler=SubsetRandomSampler(subset_indices))
        model.joint_training(MIN_FAMILIARITY_THRESHOLD)


# %%

def predict_classifier(clf):
    print("Making predictions")
    test_loss = 0
    correct = 0
    clf.eval()
    for batch_idx, (data, target) in enumerate(test_dataset_loader):
        data = data.to(clf.device)
        target = target.to(clf.device)
        out = clf(data)
        test_loss += clf.loss_function(out, target.long()).item()
        pred = out.data.max(1)[1]
        # target_pred = target.data.max(1)[1]
        # correct += pred.eq(target_pred).sum()
        correct += pred.eq(target.data).sum()
    test_loss /= len(test_dataset_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_dataset_loader.dataset),
        100. * correct / len(test_dataset_loader.dataset)))


def train_classifier(clf):
    clf.train()
    for epoch in range(20):
        for batch_idx, (data, target) in enumerate(train_dataset_loader):
            data = data.to(clf.device)
            target = target.to(clf.device)
            optimizer.zero_grad()
            out = clf(data)
            loss = clf.loss_function(out, target.long())
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_dataset_loader.dataset),
                           100.0 * batch_idx / len(train_dataset_loader), loss.item()))
        predict_classifier(clf)


# %% Convert the training set to the unsupervised latent vector
print("Converting images to latent vectors")
classifier_training_batch_size = 1
train_loader = torch.utils.data.DataLoader(train_data, batch_size=classifier_training_batch_size, shuffle=False)
counter = 0
training_features = []
training_labels = []
new_dataset = []
for batch_idx, (data, target) in enumerate(train_loader):
    data = data.to(model.device)
    data_cropped_center = crop(data, int(image_size / 2) - rbm_input_size, int(image_size / 2) - rbm_input_size,
                               rbm_input_size, rbm_input_size)
    regions_to_check = [
        crop(data, int(image_size / 2) - rbm_input_size - rbm_input_size,
             int(image_size / 2) - rbm_input_size - rbm_input_size, rbm_input_size, rbm_input_size),
        crop(data, int(image_size / 2) - rbm_input_size + rbm_input_size,
             int(image_size / 2) - rbm_input_size + rbm_input_size, rbm_input_size, rbm_input_size),
        crop(data, int(image_size / 2) - rbm_input_size - rbm_input_size,
             int(image_size / 2) - rbm_input_size + rbm_input_size, rbm_input_size, rbm_input_size),
        crop(data, int(image_size / 2) - rbm_input_size + rbm_input_size,
             int(image_size / 2) - rbm_input_size - rbm_input_size, rbm_input_size, rbm_input_size),
    ]
    latent_vector = []
    for m in model.models:
        # Encode the image
        rbm_input = m.encode(data_cropped_center)
        # Flatten input for RBM
        flat_rbm_input = rbm_input.view(len(rbm_input), RBM_VISIBLE_UNITS)

        # Compare data with existing models
        familiriaty = m.rbm.is_familiar(flat_rbm_input, provide_value=False)
        # values = values.cpu().detach().numpy()
        #
        # latent_vector.append(values)

        for m_child in m.child_networks:
            for region in regions_to_check:
                if familiriaty == 0:
                    latent_vector.append(0)
                    continue
                # Encode the image
                rbm_input = m_child.encode(region)
                # Flatten input for RBM
                flat_rbm_input = rbm_input.view(len(rbm_input), RBM_VISIBLE_UNITS)

                # Compare data with existing models
                values = m_child.rbm.is_familiar(flat_rbm_input)
                values = values.cpu().detach().numpy()

                latent_vector.append(values[0])

    latent_vector = np.array(latent_vector)
    target_labels = target.cpu().detach().numpy()
    for i in range(classifier_training_batch_size):
        # training_features.append(latent_vector[:, i])
        training_features.append(latent_vector)
        training_labels.append(target_labels[i])

    counter += 1
    if counter % 100 == 0:
        print("Training iteration: ", counter)
# new_dataset = np.array(new_dataset, dtype=float)
training_features = np.array(training_features)
training_features_norm = preprocessing.scale(training_features)
training_labels = np.array(training_labels, dtype=float)
# %%
train_dataset = UnsupervisedVectorDataset(training_features, training_labels)
train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)

# %%
print("Converting test images to latent vectors")
test_batch_size = 1
test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
counter = 0
test_features = []
test_labels = []
for batch_idx, (data, target) in enumerate(test_loader):
    data = data.to(model.device)
    data_cropped_center = crop(data, int(image_size / 2) - rbm_input_size, int(image_size / 2) - rbm_input_size,
                               rbm_input_size, rbm_input_size)
    regions_to_check = [
        crop(data, int(image_size / 2) - rbm_input_size - rbm_input_size,
             int(image_size / 2) - rbm_input_size - rbm_input_size, rbm_input_size, rbm_input_size),
        crop(data, int(image_size / 2) - rbm_input_size + rbm_input_size,
             int(image_size / 2) - rbm_input_size + rbm_input_size, rbm_input_size, rbm_input_size),
        crop(data, int(image_size / 2) - rbm_input_size - rbm_input_size,
             int(image_size / 2) - rbm_input_size + rbm_input_size, rbm_input_size, rbm_input_size),
        crop(data, int(image_size / 2) - rbm_input_size + rbm_input_size,
             int(image_size / 2) - rbm_input_size - rbm_input_size, rbm_input_size, rbm_input_size),
    ]
    latent_vector = []
    for m in model.models:
        # Encode the image
        rbm_input = m.encode(data_cropped_center)
        # Flatten input for RBM
        flat_rbm_input = rbm_input.view(len(rbm_input), RBM_VISIBLE_UNITS)

        familiriaty = m.rbm.is_familiar(flat_rbm_input, provide_value=False)
        #
        # # Compare data with existing models
        # values = m.rbm.is_familiar(flat_rbm_input)
        # values = values.cpu().detach().numpy()
        #
        # latent_vector.append(values)

        for m_child in m.child_networks:
            for region in regions_to_check:
                if familiriaty == 0:
                    latent_vector.append(0)
                    continue
                # Encode the image
                rbm_input = m_child.encode(region)
                # Flatten input for RBM
                flat_rbm_input = rbm_input.view(len(rbm_input), RBM_VISIBLE_UNITS)

                # Compare data with existing models
                values = m_child.rbm.is_familiar(flat_rbm_input)
                values = values.cpu().detach().numpy()

                latent_vector.append(values[0])

    latent_vector = np.array(latent_vector)
    target_labels = target.cpu().detach().numpy()
    for i in range(test_batch_size):
        # test_features.append(latent_vector[:, i])
        test_features.append(latent_vector)
        test_labels.append(target_labels[i])
    counter += 1
    if counter % 100 == 0:
        print("Testing iteration: ", counter)

test_features = np.array(test_features)
test_features_norm = preprocessing.scale(test_features)
test_labels = np.array(test_labels)
# %%
test_dataset = UnsupervisedVectorDataset(test_features, test_labels)
test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

# %% Training svm
# from sklearn.svm import LinearSVC
#
# clf = LinearSVC(tol=1e-5)
#
# clf.fit(training_features_norm, training_labels)
# predictions = clf.predict(test_features_norm)
# print('Result: %d/%d' % (sum(predictions == test_labels), test_labels.shape[0]))


# %% Training classifier
print("Training classifier")
clf = Classifier(training_features_norm.shape[1])
# criterion = nn.NLLLoss()
# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3, amsgrad=True)

train_classifier(clf)
