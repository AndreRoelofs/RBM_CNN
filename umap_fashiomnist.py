import numpy as np
import umap
import matplotlib.pyplot as plt

classes = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot']

# classes = [
#     'Airplane',
#     'Automobile',
#     'Bird',
#     'Cat',
#     'Deer',
#     'Dog',
#     'Frog',
#     'Horse',
#     'Ship',
#     'Truck']

# model_type = 'simple'
# model_type = 'large'
# model_type = 'CIFAR_10_rbm_fixed_5'
# model_type = 'rbm_fixed_5'
# model_type = 'CIFAR_10_large_rbm_fixed_3'
model_type = 'large_rbm_fixed_3'
# model_type = 'large_fixed'
# model_type = 'sequential'
n_levels = 1

train_features = np.load('one_layered_wdn/{}_level_train_features_{}.npy'.format(n_levels, model_type))
train_labels = np.load('one_layered_wdn/{}_level_train_labels_{}.npy'.format(n_levels, model_type))

# train_features = np.load('autoencoder/fashion_mnist_train_features_ae_392.npy'.format(n_levels, model_type))
# train_labels = np.load('autoencoder/fashion_mnist_train_labels_ae_392.npy'.format(n_levels, model_type))

train_features -= train_features.min(0)
train_features /= train_features.max(0)

# data = np.array(np.vstack([train_features, test_features]), dtype=np.float64)

# embedding = umap.UMAP(n_neighbors=10).fit_transform(train_features)
embedding = umap.UMAP().fit_transform(train_features)

fig, ax = plt.subplots(1, figsize=(14, 10))
plt.scatter(*embedding.T, s=0.3, c=train_labels, cmap='Spectral', alpha=1.0)
plt.setp(ax, xticks=[], yticks=[])
cbar = plt.colorbar(boundaries=np.arange(11) - 0.5)
cbar.set_ticks(np.arange(10))
cbar.set_ticklabels(classes)
plt.title('Fashion MNIST Embedded via UMAP')
# plt.title('CIFAR 10 Embedded via UMAP')
plt.show()
