import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
import torchvision.datasets
import torchvision.models
from torchvision import transforms
import matplotlib.pyplot as plt

from rbm_example.rbm import RBM
from rbm_example.rv_rbm import RV_RBM

########## CONFIGURATION ##########
BATCH_SIZE = 64
VISIBLE_UNITS = 784  # 28 x 28 images
HIDDEN_UNITS = 128
CD_K = 1
EPOCHS = 10

DATA_FOLDER = 'data'

CUDA = torch.cuda.is_available()
CUDA_DEVICE = 0

if CUDA:
    torch.cuda.set_device(CUDA_DEVICE)

########## LOADING DATASET ##########
print('Loading dataset...')

train_dataset = torchvision.datasets.FashionMNIST(root=DATA_FOLDER, train=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
]), download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)

test_dataset = torchvision.datasets.FashionMNIST(root=DATA_FOLDER, train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
]), download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

########## TRAINING RBM ##########
print('Training RBM...')

# rbm = RBM(VISIBLE_UNITS, HIDDEN_UNITS, CD_K, use_cuda=CUDA)
rbm = RV_RBM(VISIBLE_UNITS, HIDDEN_UNITS, 1, CD_K, use_cuda=CUDA, use_relu=True)

for epoch in range(EPOCHS):
    epoch_error = 0.0

    for batch, _ in train_loader:
        batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data

        if CUDA:
            batch = batch.cuda()

        batch_error = rbm.contrastive_divergence(batch)

        epoch_error += batch_error

    print('Epoch Error (epoch=%d): %.4f' % (epoch, epoch_error))

########## EXTRACT FEATURES ##########
print('Extracting features...')

train_features = np.zeros((len(train_dataset), HIDDEN_UNITS))
train_labels = np.zeros(len(train_dataset))
test_features = np.zeros((len(test_dataset), HIDDEN_UNITS))
test_labels = np.zeros(len(test_dataset))

for i, (batch, labels) in enumerate(train_loader):
    batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data

    if CUDA:
        batch = batch.cuda()

    train_features[i * BATCH_SIZE:i * BATCH_SIZE + len(batch)] = rbm.sample_hidden(batch).cpu().numpy()
    train_labels[i * BATCH_SIZE:i * BATCH_SIZE + len(batch)] = labels.numpy()

for i, (batch, labels) in enumerate(test_loader):
    batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data

    if CUDA:
        batch = batch.cuda()

    hidden = rbm.sample_hidden(batch)

    recon_image = rbm.sample_visible(hidden)

    index = 1
    plt.imshow(batch[index].cpu().detach().numpy().reshape((28, 28)), cmap='gray')
    plt.show()

    plt.imshow(recon_image[index].reshape((28, 28)).cpu().detach().numpy(), cmap='gray')
    plt.show()

    break

    test_features[i * BATCH_SIZE:i * BATCH_SIZE + len(batch)] = hidden.cpu().numpy()
    test_labels[i * BATCH_SIZE:i * BATCH_SIZE + len(batch)] = labels.numpy()

########## RECONSTRUCTION ##########


########## CLASSIFICATION ##########
print('Classifying...')

clf = LogisticRegression()
clf.fit(train_features, train_labels)
predictions = clf.predict(test_features)

print('Result: %d/%d' % (sum(predictions == test_labels), test_labels.shape[0]))
