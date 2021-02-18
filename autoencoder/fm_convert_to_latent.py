from autoencoder.model import Autoencoder
import numpy as np
import torch
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torchvision import transforms
from one_layered_wdn.main import train_knn, print_cluster_ids

data_path = '../data'

latent_vector_size = 392
batch_size = 100
n_clusters = 80

train_size = 60000
test_size = 10000

# train_size = 100
# test_size = 100

# model = Autoencoder()
model = torch.load('fm_ae_checkpoint/model_best.pth.tar')
# checkpoint =
# model.load_state_dict(checkpoint)

device = torch.device('cuda:0')
model.to(device)

train_data = FashionMNIST(data_path, train=True, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                          ]))

test_data = FashionMNIST(data_path, train=False, transform=transforms.Compose([
    transforms.ToTensor(),
]))

train_data.data = train_data.data[:train_size]
train_data.targets = train_data.targets[:train_size]

test_data.data = test_data.data[:test_size]
test_data.targets = test_data.targets[:test_size]

train_features = np.zeros((train_size, latent_vector_size))
train_labels = np.zeros(train_size)

test_features = np.zeros((test_size, latent_vector_size))
test_labels = np.zeros(test_size)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

counter = 0
for step, (images_raw, labels) in enumerate(train_loader):
    images_raw = images_raw.to(device)
    p, _ = model.encode(images_raw.float())
    p = torch.flatten(p, start_dim=1)
    train_features[counter:counter + batch_size] = p.cpu().detach().numpy()
    train_labels[counter:counter + batch_size] = labels.numpy()

    counter += batch_size

counter = 0
for step, (images_raw, labels) in enumerate(test_loader):
    images_raw = images_raw.to(device)
    p, _ = model.encode(images_raw.float())
    p = torch.flatten(p, start_dim=1)
    test_features[counter:counter + batch_size] = p.cpu().detach().numpy()
    test_labels[counter:counter + batch_size] = labels.numpy()

    counter += batch_size

np.save('fashion_mnist_train_features_ae_{}.npy'.format(latent_vector_size), train_features)
np.save('fashion_mnist_train_labels_ae_{}.npy'.format(latent_vector_size), train_labels)

np.save('fashion_mnist_test_features_ae_{}.npy'.format(latent_vector_size), test_features)
np.save('fashion_mnist_test_labels_ae_{}.npy'.format(latent_vector_size), test_labels)

print("Fit KNN")
cluster_ids_x, cluster_ids_y = train_knn(train_features, test_features, n_clusters)

np.save('fashion_mnist_ae_{}_train_clusters_{}.npy'.format(latent_vector_size, n_clusters), cluster_ids_x)
np.save('fashion_mnist_ae_{}_test_clusters_{}.npy'.format(latent_vector_size, n_clusters), cluster_ids_y)
#
# cluster_ids_x = np.load('{}_level_train_clusters_{}_{}.npy'.format(n_levels, n_clusters, model_type))
# cluster_ids_y = np.load('{}_level_test_clusters_{}_{}.npy'.format(n_levels, n_clusters, model_type))

train_bins = print_cluster_ids(cluster_ids_x, train_labels, n_clusters)
print("________________")
test_bins = print_cluster_ids(cluster_ids_y, test_labels, n_clusters)

error_counter = 0
error_pos = []
for i in range(train_bins.shape[0]):
    tr_bin = train_bins[i]
    te_bin = test_bins[i]

    for j in range(tr_bin.shape[0]):
        if tr_bin[j] == 0 and te_bin[j] > 0:
            error_counter += te_bin[j]
            error_pos.append([i, j])
print("Errors: {}".format(error_counter))
for pos in error_pos:
    print("Error bin: {} class: {}".format(pos[0], pos[1]))

test = 0
