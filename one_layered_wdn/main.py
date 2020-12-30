import numpy as np
import torch
from sklearn.svm import SVC, LinearSVC
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torch.utils.data.sampler import SubsetRandomSampler
import configparser
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from one_layered_wdn.helpers import *
from one_layered_wdn.custom_transforms import CropBlackPixelsAndResize
from one_layered_wdn.wdn import WDN
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from one_layered_wdn.custom_dataset import UnsupervisedVectorDataset
from one_layered_wdn.custom_classifier import FullyConnectedClassifier, train_classifier

# General
config = None
data_path = '../data'

# Encoder settings
input_filters = None
encoder_output_filters = None
encoder_activation = None  # Activation function
encoder_learning_rate = None

# RBM settings
image_input_size = None
rbm_visible_units = None
rbm_hidden_units = None
rbm_learning_rate = None
rbm_activation = None

# Training settings
node_train_batch_size = None
min_familiarity_threshold = None
train_data = None
test_data = None
fast_training = None
fastest_training = None
tolerance = 0.5


def process_settings():
    # Setup dataset to use
    global input_filters
    global fast_training
    global fastest_training

    general_settings = config['GENERAL']
    if general_settings['Dataset'] == MNIST_DATASET:
        input_filters = 1
    if general_settings['Dataset'] == FASHIONMNIST_DATASET:
        input_filters = 1
    if general_settings['Dataset'] == CIFAR10_DATASET:
        input_filters = 3

    fast_training = general_settings['FastTraining'] == 'True'
    fastest_training = general_settings['FastestTraining'] == 'True'

    # Setup Encoder
    global encoder_activation
    global encoder_output_filters
    global encoder_learning_rate

    encoder_settings = config['ENCODER']
    encoder_activation = encoder_settings['ActivationFunction']
    encoder_output_filters = int(encoder_settings['NumberOfFilters'])
    encoder_learning_rate = float(encoder_settings['LearningRate'])

    # Setup RBM
    global image_input_size
    global rbm_visible_units
    global rbm_hidden_units
    global rbm_activation
    global rbm_learning_rate

    rbm_settings = config['RBM']
    image_input_size = int(rbm_settings['ImageInputSize'])
    rbm_visible_units = encoder_output_filters * image_input_size ** 2
    rbm_hidden_units = int(rbm_settings['NumberOfHiddenUnits'])
    rbm_activation = rbm_settings['ActivationFunction']
    rbm_learning_rate = float(rbm_settings['LearningRate'])

    # Setup Training
    global node_train_batch_size
    global min_familiarity_threshold

    node_training_settings = config['NODE_TRAINING']
    node_train_batch_size = int(node_training_settings['TrainBatchSize'])
    min_familiarity_threshold = int(node_training_settings['MinFamiliarityThreshold'])


def load_data():
    global train_data
    global test_data
    global tolerance
    general_settings = config['GENERAL']
    if general_settings['Dataset'] == MNIST_DATASET:
        train_data = MNIST(data_path, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               # CropBlackPixelsAndResize(tol=tolerance, output_size=image_input_size),
                               transforms.Resize((image_input_size, image_input_size)),
                           ]))

        test_data = MNIST(data_path, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            # CropBlackPixelsAndResize(tol=tolerance, output_size=image_input_size),
            transforms.Resize((image_input_size, image_input_size)),
        ]))

    if general_settings['Dataset'] == FASHIONMNIST_DATASET:
        train_data = FashionMNIST(data_path, train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      CropBlackPixelsAndResize(tol=tolerance, output_size=image_input_size),
                                  ]))

        test_data = FashionMNIST(data_path, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            CropBlackPixelsAndResize(tol=tolerance, output_size=image_input_size),
        ]))
    if general_settings['Dataset'] == CIFAR10_DATASET:
        train_data = CIFAR10(data_path, train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                             ]))

        test_data = CIFAR10(data_path, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ]))

    if fast_training:
        train_data.data = train_data.data[:10000]
        train_data.targets = train_data.targets[:10000]

        test_data.data = test_data.data[:1000]
        test_data.targets = test_data.targets[:1000]

    if fastest_training:
        train_data.data = train_data.data[:1000]
        train_data.targets = train_data.targets[:1000]

        test_data.data = test_data.data[:100]
        test_data.targets = test_data.targets[:100]



if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    config = configparser.ConfigParser()
    config.read('config.ini')

    process_settings()

    load_data()

    node_settings = {
        'image_input_size': image_input_size,
        'image_channels': input_filters,

        'encoder_channels': encoder_output_filters,
        'encoder_learning_rate': encoder_learning_rate,
        'encoder_activation': encoder_activation,

        'rbm_visible_units': rbm_visible_units,
        'rbm_hidden_units': rbm_hidden_units,
        'rbm_learning_rate': rbm_learning_rate,
        'rbm_activation': rbm_activation,

        'min_familiarity_threshold': min_familiarity_threshold

    }

    model = WDN(node_settings)

    for i in range(10):
    # for i in [5]:
        print("Training digit: ", i)
        subset_indices = (torch.tensor(train_data.targets) == i).nonzero().view(-1)
        subset_indices = subset_indices[torch.randperm(subset_indices.size()[0])]
        model.train_loader = torch.utils.data.DataLoader(train_data, batch_size=node_train_batch_size,
                                                         shuffle=False,
                                                         sampler=SubsetRandomSampler(subset_indices))
        model.joint_training()

    # exit(0)

    print("Converting train images to latent vectors")
    train_features, train_features_norm, train_labels = convert_images_to_latent_vector(train_data, model)
    print("Converting test images to latent vectors")
    test_features, test_features_norm, test_labels = convert_images_to_latent_vector(test_data, model)
    #
    train_dataset = UnsupervisedVectorDataset(train_features_norm, train_labels)
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=False)
    #
    test_dataset = UnsupervisedVectorDataset(test_features_norm, test_labels)
    test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
    #
    print("Training classifier")


    # kmeans = KMeans(n_clusters=2, random_state=0).fit(train_features)
    # predictions = kmeans.predict(test_features)
    # print('Result: %d/%d' % (sum(predictions == test_labels), train_labels.shape[0]))

    # neigh = KNeighborsClassifier(n_neighbors=1, leaf_size=1, algorithm='kd_tree', p=1, weights='distance')
    # neigh.fit(train_features, train_labels)
    # predictions = neigh.predict(test_features)
    # print('Result: %d/%d' % (sum(predictions == test_labels), test_labels.shape[0]))

    # wrong_indices = np.where(predictions != test_labels)[0]
    #
    # for i in wrong_indices:
    #     img = test_data.data[i].cpu().detach().numpy()
    #     plt.imshow(img, cmap='gray')
    #     plt.show()
    #
    fcnc = FullyConnectedClassifier(train_features_norm.shape[1])
    optimizer = torch.optim.Adam(fcnc.parameters(), lr=1e-3, amsgrad=True)

    train_classifier(fcnc, optimizer, train_dataset_loader, test_dataset_loader)

    svc = LinearSVC()
    # svc = SVC(cache_size=32768)
    svc.fit(train_features, train_labels)
    predictions = svc.predict(test_features)
    print('Result: %d/%d' % (sum(predictions == test_labels), test_labels.shape[0]))
    #
    wrong_indices = np.where(predictions != test_labels)[0]

    for i in wrong_indices:
        img = test_data.data[i].cpu().detach().numpy()
        plt.imshow(img, cmap='gray')
        plt.show()






