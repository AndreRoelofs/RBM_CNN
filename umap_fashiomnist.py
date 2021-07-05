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

# dataset = 'CIFAR_10'
dataset = 'Fashion_MNIST'

model_number = 24

train_features = np.load('one_layered_wdn/train_features_{}_old_rbm_cnn_data_normalized_quality_wide_levels_1_{}.npy'.format(dataset, model_number))
train_labels = np.load('one_layered_wdn/train_labels_{}_old_rbm_cnn_data_normalized_quality_wide_levels_1_{}.npy'.format(dataset, model_number))

# train_features = np.load('one_layered_wdn/test_features_{}_old_rbm_cnn_data_normalized_quality_wide_levels_1_18.npy'.format(dataset))
# train_labels = np.load('one_layered_wdn/test_labels_{}_old_rbm_cnn_data_normalized_quality_wide_levels_1_18.npy'.format(dataset))

train_features -= train_features.min(0)
train_features /= train_features.max(0)

# train_features -= train_features.min()
# train_features /= train_features.max()

# data = np.array(np.vstack([train_features, test_features]), dtype=np.float64)

embedding = umap.UMAP(n_neighbors=50).fit_transform(train_features)
# embedding = umap.UMAP().fit_transform(train_features)

fig, ax = plt.subplots(1, figsize=(14, 10))
plt.scatter(*embedding.T, s=0.3, c=train_labels, cmap='Spectral', alpha=1.0)
plt.setp(ax, xticks=[], yticks=[])
cbar = plt.colorbar(boundaries=np.arange(11) - 0.5)
cbar.set_ticks(np.arange(10))
cbar.set_ticklabels(classes)
plt.title('Fashion MNIST Embedded via UMAP')
# plt.title('CIFAR 10 Embedded via UMAP')
plt.show()
