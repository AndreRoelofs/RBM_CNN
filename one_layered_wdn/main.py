import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torch.utils.data.sampler import SubsetRandomSampler
import configparser
from one_layered_wdn.helpers import *
from one_layered_wdn.custom_transforms import CropBlackPixelsAndResize
from one_layered_wdn.wdn import WDN
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
    general_settings = config['GENERAL']
    if general_settings['Dataset'] == MNIST_DATASET:
        tolerance = 0.5
        train_data = MNIST(data_path, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               CropBlackPixelsAndResize(tol=tolerance, output_size=image_input_size),
                               # transforms.Resize((14, 14)),
                           ]))

        test_data = MNIST(data_path, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            CropBlackPixelsAndResize(tol=tolerance, output_size=image_input_size),
            # transforms.Resize((14, 14)),
        ]))

    if general_settings['Dataset'] == FASHIONMNIST_DATASET:
        train_data = FashionMNIST(data_path, train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                  ]))

        test_data = FashionMNIST(data_path, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
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

        test_data.data = test_data.data[:1000]
        test_data.targets = test_data.targets[:1000]



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
        print("Training digit: ", i)
        subset_indices = (torch.tensor(train_data.targets) == i).nonzero().view(-1)
        subset_indices = subset_indices[torch.randperm(subset_indices.size()[0])]
        model.train_loader = torch.utils.data.DataLoader(train_data, batch_size=node_train_batch_size,
                                                         shuffle=False,
                                                         sampler=SubsetRandomSampler(subset_indices))
        model.joint_training()

    train_features, train_features_norm, train_labels = convert_images_to_latent_vector(train_data, model)
    test_features, test_features_norm, test_labels = convert_images_to_latent_vector(test_data, model)

    train_dataset = UnsupervisedVectorDataset(train_features, train_labels)
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)

    test_dataset = UnsupervisedVectorDataset(test_features, test_labels)
    test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

    print("Training classifier")
    clf = FullyConnectedClassifier(train_features_norm.shape[1])
    # optimizer = torch.optim.SGD(clf.parameters(), lr=1e-3)
    optimizer = torch.optim.RMSprop(clf.parameters(), lr=0.0005)
    # optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3, amsgrad=True)

    train_classifier(clf, optimizer, train_dataset_loader, test_dataset_loader)




