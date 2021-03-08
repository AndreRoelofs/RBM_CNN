# Import packages
import numpy as np
import torch
from torchvision.datasets import FashionMNIST
import wandb
from one_layered_wdn.helpers import *
from torchvision import transforms
import random
from sys import exit

#%% Initialize hyperparameters
# General
data_path = "../data"
dataset_name = FASHIONMNIST_DATASET

# RBM settings
n_rbm_training = 1
n_filter_training = 2

# Dataset settings
train_data = None
test_data = None
image_size = None
input_filters = None
n_train_data = None
n_test_data = None
n_classes = None

#%% Load data
if dataset_name == FASHIONMNIST_DATASET:
    image_size = 28
    input_filters = 1
    n_train_data = 60000
    n_test_data = 10000
    n_classes = 10

    train_data = FashionMNIST(data_path, train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  # transforms.Normalize((0.1307,), (0.3081,)),
                              ]))

    test_data = FashionMNIST(data_path, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,)),
    ]))

#%% Configure Filter_RBM
wdn_settings = {
            'model_name': "test",
            'n_levels': 1,

            'image_input_size': image_size,
            'image_channels': input_filters,

            'min_familiarity_threshold': 1,
            'log_interval': 50,

            'levels_info': [
                {'input_channels': input_filters, 'encoder_channels': 1,
                 'rbm_visible_units': image_size ** 2,
                 'encoder_weight_mean': 0,
                 'encoder_weight_variance': 0.001, 'rbm_hidden_units': 300, 'rbm_learning_rate': 1e-3,
                 'encoder_learning_rate': 1e-3, 'n_training': 2},

                {'input_channels': input_filters, 'encoder_channels': 1,
                 'rbm_visible_units': int(image_size / 2) ** 2,
                 'encoder_weight_variance': 0.07, 'rbm_hidden_units': 100, 'rbm_learning_rate': 1e-3,
                 'encoder_learning_rate': 1e-3, 'n_training': 2},

                {'input_channels': input_filters, 'encoder_channels': 1,
                 'rbm_visible_units': int(image_size / 4) ** 2,
                 'encoder_weight_variance': 0.07, 'rbm_hidden_units': 50, 'rbm_learning_rate': 1e-3,
                 'encoder_learning_rate': 1e-3, 'n_training': 2},
            ]
        }



#%% Create random node
reset_seed()

random_tr_indice = np.random.randint(0, n_train_data)

image = train_data.data[random_tr_indice]
label = train_data.targets[random_tr_indice]







