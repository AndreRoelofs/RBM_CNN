import numpy as np
import torch
from sklearn.svm import SVC, LinearSVC
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, FashionMNIST
from torch.utils.data.sampler import SubsetRandomSampler
import configparser
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from one_layered_wdn.helpers import *
from one_layered_wdn.custom_transforms import CropBlackPixelsAndResize
from one_layered_wdn.wdn import WDN, train_wdn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from one_layered_wdn.custom_dataset import UnsupervisedVectorDataset
import random
from one_layered_wdn.custom_classifier import FullyConnectedClassifier, train_classifier, predict_classifier, FashionCNN
import one_layered_wdn.svm as svm
from torch import nn
# from one_layered_wdn.kmeans import kmeans, kmeans_predict
from kmeans_pytorch import kmeans, kmeans_predict
import scipy.cluster.hierarchy as hcluster
import copy
from ImbalancedDatasetSampler import ImbalancedDatasetSampler
from sys import exit

import wandb

# General
config = None
data_path = '../data'

# Encoder settings
input_filters = None
encoder_output_filters = None
encoder_activation = None  # Activation function
encoder_learning_rate = None

# RBM settings
image_size = None
rbm_visible_units = None
rbm_hidden_units = None
rbm_learning_rate = None
rbm_activation = None

# Training settings
node_train_batch_size = None
min_familiarity_threshold = None
train_data = None
val_data = None
test_data = None
fast_training = None
fastest_training = None
tolerance = 0.5
big_cnnc = None


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
        input_filters = 1
    if general_settings['Dataset'] == CIFAR100_DATASET:
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
    global image_size
    global rbm_visible_units
    global rbm_hidden_units
    global rbm_activation
    global rbm_learning_rate

    rbm_settings = config['RBM']
    image_size = int(rbm_settings['ImageInputSize'])
    rbm_visible_units = encoder_output_filters * image_size ** 2
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
    global val_data
    global test_data
    global tolerance

    general_settings = config['GENERAL']
    if general_settings['Dataset'] == MNIST_DATASET:
        train_data = MNIST(data_path, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               # transforms.Normalize((0.1307,), (0.3081,)),
                               # CropBlackPixelsAndResize(tol=tolerance, output_size=image_input_size),
                               # transforms.Resize((image_input_size, image_input_size)),
                           ]))

        test_data = MNIST(data_path, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,)),
            # CropBlackPixelsAndResize(tol=tolerance, output_size=image_input_size),
            # transforms.Resize((image_input_size, image_input_size)),
        ]))

    if general_settings['Dataset'] == FASHIONMNIST_DATASET:
        train_data = FashionMNIST(data_path, train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,)),
                                  ]))

        val_data = FashionMNIST(data_path, train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,)),
                                  ]))

        test_data = FashionMNIST(data_path, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]))

    if general_settings['Dataset'] == CIFAR10_DATASET:
        train_data = CIFAR10(data_path, train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                 #
                                 transforms.Grayscale(),
                                 transforms.Normalize((0.1307,), (0.3081,)),
                             ]))

        test_data = CIFAR10(data_path, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

            transforms.Grayscale(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]))

    if general_settings['Dataset'] == CIFAR100_DATASET:
        train_data = CIFAR100(data_path, train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                  # transforms.Grayscale(),
                              ]))

        test_data = CIFAR100(data_path, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Grayscale(),
        ]))

    if fast_training:
        np.random.seed(0)
        train_indices = np.random.randint(0, train_data.data.shape[0], 10000)
        train_data.data = train_data.data[train_indices]
        train_data.targets = np.array(train_data.targets)[train_indices]

        test_indices = np.random.randint(0, test_data.data.shape[0], 1000)
        test_data.data = test_data.data[test_indices]
        test_data.targets = np.array(test_data.targets)[test_indices]

    if fastest_training:
        np.random.seed(0)
        train_indices = np.random.randint(0, train_data.data.shape[0], 1000)
        train_data.data = train_data.data[train_indices]
        train_data.targets = np.array(train_data.targets)[train_indices]

        test_indices = np.random.randint(0, test_data.data.shape[0], 100)
        test_data.data = test_data.data[test_indices]
        test_data.targets = np.array(test_data.targets)[test_indices]


def calculate_average_accuracy_over_clusters(train_predictions, test_predictions, n_clusters):
    reset_seed()

    global big_cnnc

    accuracies = []
    if big_cnnc is None:
        cluster_cnn_train_dataloader = torch.utils.data.DataLoader(train_data,
                                                                   batch_size=100,
                                                                   shuffle=True,
                                                                   )
        cluster_cnn_test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False,
                                                                  )
        big_cnnc = FashionCNN()
        cnnc_optimizer = torch.optim.Adam(big_cnnc.parameters(), lr=1e-3)
        train_classifier(big_cnnc, cnnc_optimizer, cluster_cnn_train_dataloader, cluster_cnn_test_dataloader, [], 10)
        for param in big_cnnc.parameters():
            param.requires_grad = False
        del cluster_cnn_train_dataloader
        del cluster_cnn_test_dataloader

    for cluster_id in range(n_clusters):
        # for cluster_id in range(0, 1):
        print("Current cluster ", cluster_id)
        train_cluster_idx = []
        for i in range(len(train_predictions)):
            cluster = train_predictions[i]
            if cluster != cluster_id:
                continue
            train_cluster_idx.append(i)

        cluster_cnn_train_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=min(128, len(train_cluster_idx)),
            # batch_size=len(train_cluster_idx),
            shuffle=False,
            sampler=ImbalancedDatasetSampler(dataset=train_data, indices=train_cluster_idx),
            # sampler=SubsetRandomSampler(train_cluster_idx)
        )
        test_cluster_idx = []
        for i in range(len(test_predictions)):
            cluster = test_predictions[i]
            if cluster != cluster_id:
                continue
            test_cluster_idx.append(i)

        # for idx in test_cluster_idx[58:59]:
        #     plt.imshow(test_data.data[idx].reshape((28, 28)).cpu().detach().numpy(), cmap='gray')
        #     plt.show()
        #
        # return

        cluster_cnn_test_dataloader = torch.utils.data.DataLoader(
            test_data,
            batch_size=min(100, len(test_cluster_idx)),
            shuffle=False,
            sampler=SubsetRandomSampler(test_cluster_idx)
        )
        big_cnnc_clone = FashionCNN()
        big_cnnc_clone.load_state_dict(copy.deepcopy(big_cnnc.state_dict()))
        for param in big_cnnc_clone.parameters():
            param.requires_grad = True

        cnnc_optimizer = torch.optim.Adam(big_cnnc_clone.parameters(), lr=1e-4)
        # cnnc_optimizer = torch.optim.SGD(big_cnnc_clone.parameters(), lr=1e-3)
        train_classifier(big_cnnc_clone, cnnc_optimizer, cluster_cnn_train_dataloader, cluster_cnn_test_dataloader,
                         accuracies, 100)
        print("General Classifier:")
        predict_classifier(big_cnnc, cluster_cnn_test_dataloader, [])

        big_cnnc_clone.cpu()
        del big_cnnc_clone
        del cluster_cnn_train_dataloader
        del cluster_cnn_test_dataloader

    print("Average accuracy over {} clusters is {}".format(n_clusters, np.sum(accuracies)))
    print(accuracies)


# def train_knn(train_features, val_features, test_features, n_clusters):
#     device = torch.device('cpu')
#     tr_features = torch.tensor(train_features, dtype=torch.float)
#     va_features = torch.tensor(val_features, dtype=torch.float)
#     te_features = torch.tensor(test_features, dtype=torch.float)
#
#     cluster_ids_x, cluster_centers = kmeans(
#         X=tr_features, num_clusters=n_clusters,
#         distance='euclidean',
#         device=device
#     )
#     cluster_ids_val = kmeans_predict(va_features, cluster_centers,
#                                    distance='euclidean',
#                                    device=device)
#
#     cluster_ids_y = kmeans_predict(te_features, cluster_centers,
#                                    distance='euclidean',
#                                    device=device)
#
#     return cluster_ids_x, cluster_ids_val, cluster_ids_y

def train_knn(train_features, test_features, n_clusters):
    device = torch.device('cpu')
    tr_features = torch.tensor(train_features, dtype=torch.float)

    # tr_features -= tr_features.min(0)[0]
    # tr_features /= tr_features.max(0)[0]

    # tr_features -= tr_features.min()
    # tr_features /= tr_features.max()

    te_features = torch.tensor(test_features, dtype=torch.float)

    # te_features -= te_features.min(0)[0]
    # te_features /= te_features.max(0)[0]
    #
    # te_features -= te_features.min()
    # te_features /= te_features.max()


    cluster_ids_x, cluster_centers = kmeans(
        X=tr_features, num_clusters=n_clusters,
        # tol=1e-6,
        distance='euclidean',
        # distance='cosine',
        device=device
    )
    cluster_ids_y = kmeans_predict(te_features, cluster_centers,
                                   distance='euclidean',
                                   # distance='cosine',
                                   device=device)

    return cluster_ids_x, cluster_ids_y

def calculate_max_clusters(train_features, test_features):
    cluster_ids_x = np.zeros(train_features.shape[0], dtype=np.int)
    for i in range(train_features.shape[0]):
        cluster_ids_x[i] = train_features[i].argmax()

    # Detect empty clusters
    used_clusters = []
    for i in range(cluster_ids_x.max() + 1):
        if i in cluster_ids_x:
            used_clusters.append(i)

    cluster_counter = 0
    for cluster_id in used_clusters:
        cluster_ids_x = np.where(cluster_ids_x == cluster_id, cluster_counter, cluster_ids_x)
        cluster_counter += 1

    cluster_ids_y = np.zeros(test_features.shape[0], dtype=np.int)
    for i in range(test_features.shape[0]):
        cluster_ids_y[i] = test_features[i][used_clusters].argmax()

    cluster_counter = 0
    for cluster_id in used_clusters:
        cluster_ids_y = np.where(cluster_ids_y == cluster_id, cluster_counter, cluster_ids_y)
        cluster_counter += 1

    return cluster_ids_x, cluster_ids_y, len(used_clusters)


def calculate_cluster_bins(cluster_ids, data_labels, n_clusters, n_classes):
    bins = np.zeros((n_clusters, 10))
    for i in range(len(cluster_ids)):
        cluster = cluster_ids[i]
        bins[cluster][int(data_labels[i])] += 1
    return bins


def print_cluster_ids(bins):
    bin_counter = 0
    for bin in bins:
        bin_string = ''
        for amount in bin:
            a_size = len(str(int(amount)))
            for i in range(a_size, 6):
                bin_string += ' '
            bin_string += str(int(amount))
        print(bin_counter, bin_string)
        bin_counter += 1


def calculate_equal_clusters(bins):
    class_dis = []
    for bin in bins:
        class_dis.append([i for i, e in enumerate(bin) if e != 0])
    indices_to_skip = set()
    equal_clusters = []
    counter = n_clusters
    for i in range(len(class_dis)):
        if i in indices_to_skip:
            continue
        parent_cluster = [i]
        set_1 = set(class_dis[i])
        for j in range(i + 1, len(class_dis)):
            if j in indices_to_skip:
                continue
            if set_1 == set(class_dis[j]):
                indices_to_skip.add(j)
                indices_to_skip.add(i)

                parent_cluster.append(j)

                counter -= 1
        equal_clusters.append(parent_cluster)
    return equal_clusters

def is_slice_in_list(s,l):
    len_s = len(s) #so we don't recompute length of s on every iteration
    return any(s == l[i:len_s+i] for i in range(len(l) - len_s+1))

def compress_clusters(cluster_ids, clusters):
    for i in range(len(clusters)):
        cluster_group = clusters[i]

        for cluster in cluster_group:
            if cluster not in cluster_ids:
                continue
            cluster_ids[cluster_ids == cluster] = i

    return cluster_ids


if __name__ == "__main__":
    # reset_seed()
    config = configparser.ConfigParser()
    config.read('config.ini')

    process_settings()

    load_data()
    n_clusters = 80
    n_levels = 1
    n_classes = 10
    # model_name = '{}_rbm_cnn_finetuned_levels_{}'.format(config['GENERAL']['Dataset'] + '_old', n_levels)
    model_name = '{}_rbm_cnn_data_normalized_quality_wide_levels_{}'.format(config['GENERAL']['Dataset'] + '_old', n_levels)
    # model_name = '{}_rbm_cnn_data_normalized_quality_wide_levels_{}'.format(config['GENERAL']['Dataset'] + '_old_val', n_levels)
    for model_number in [21]:
        wdn_settings = {
            'model_name': model_name,
            'n_clusters': n_clusters,
            'n_levels': n_levels,

            'image_input_size': image_size,
            'image_channels': input_filters,

            'use_relu': False,

            'min_familiarity_threshold': min_familiarity_threshold,
            'log_interval': 50,

            'levels_info': [
                {
                    'input_channels': input_filters, 'encoder_channels': 1, 'rbm_visible_units': image_size ** 2,
                    'encoder_weight_mean': 0.1, 'encoder_weight_variance': 0.01,
                    'rbm_weight_mean': 0.0, 'rbm_weight_variance': 0.01,
                    'rbm_hidden_units': 300, 'encoder_learning_rate': 1e-3,
                    'n_training': 50, 'n_training_second': 11,
                },
            ]
        }
        wbc = None
        # wandb.init(project=model_name, config=wdn_settings, reinit=True)
        #
        # wbc = wandb.config
        # n_test_data = 5000
        # test_subset = np.random.randint(0, 10000, n_test_data)
        # test_data.data = test_data.data[test_subset]
        # test_data.targets = test_data.targets[test_subset]

        # indices = np.load('../random_erasing/fashion_mnist_training_indices.npy')
        # train_indices = indices[:50000]
        # val_indices = indices[50000:]
        #
        # train_data.data = train_data.data[train_indices]
        # train_data.targets = train_data.targets[train_indices]
        #
        # val_data.data = val_data.data[val_indices]
        # val_data.targets = val_data.targets[val_indices]
        #


        print("Train WDN")
        model = train_wdn(train_data, test_data, wdn_settings, wbc)
        print("Convert train images to latent vectors")
        train_features, _, train_labels = convert_images_to_latent_vector(train_data, model)

        # print("Convert val images to latent vectors")
        # val_features, _, val_labels = convert_images_to_latent_vector(val_data, model)

        print("Convert test images to latent vectors")
        test_features, _, test_labels = convert_images_to_latent_vector(test_data, model)

        np.save('train_features_{}_{}'.format(model_name, model_number), train_features)
        np.save('train_labels_{}_{}'.format(model_name, model_number), train_labels)
        #
        # np.save('val_features_{}_{}'.format(model_name, model_number), val_features)
        # np.save('val_labels_{}_{}'.format(model_name, model_number), val_labels)

        np.save('test_features_{}_{}'.format(model_name, model_number), test_features)
        np.save('test_labels_{}_{}'.format(model_name, model_number), test_labels)

        train_features = np.load('train_features_{}_{}.npy'.format(model_name, model_number))
        train_labels = np.load('train_labels_{}_{}.npy'.format(model_name, model_number))
        #
        # val_features = np.load('val_features_{}_{}.npy'.format(model_name, model_number))
        # val_labels = np.load('val_labels_{}_{}.npy'.format(model_name, model_number))

        test_features = np.load('test_features_{}_{}.npy'.format(model_name, model_number))
        test_labels = np.load('test_labels_{}_{}.npy'.format(model_name, model_number))

        # print("Fitting SVM")
        # svc = LinearSVC(max_iter=100000, loss='hinge', random_state=0)
        # svc = SVC(cache_size=32768, tol=1e-5, kernel='linear', random_state=0)
        # svc.fit(train_features, train_labels)
        # print("Predicting SVM")
        # predictions = svc.predict(train_features)
        # print('Train Result: %d/%d' % (np.sum(predictions == train_labels), train_labels.shape[0]))
        # predictions = svc.predict(test_features)
        # print('Test Result: %d/%d' % (np.sum(predictions == test_labels), test_labels.shape[0]))
        #
        # exit(1)
        #
        # print("Calculate Max RBM")
        # cluster_ids_x, cluster_ids_y, n_clusters = calculate_max_clusters(train_features, test_features)
        print("Fit KNN")
        cluster_ids_x, cluster_ids_y = train_knn(train_features, test_features, n_clusters)

        # cluster_ids_x, cluster_ids_val, cluster_ids_y = train_knn(train_features, val_features, test_features, n_clusters)
        #
        np.save('train_clusters_{}_{}_{}.npy'.format(model_name, model_number, n_clusters), cluster_ids_x)

        # np.save('val_clusters_{}_{}_{}.npy'.format(model_name, model_number, n_clusters), cluster_ids_val)

        np.save('test_clusters_{}_{}_{}.npy'.format(model_name, model_number, n_clusters), cluster_ids_y)

        cluster_ids_x = np.load('train_clusters_{}_{}_{}.npy'.format(model_name, model_number, n_clusters))
        # cluster_ids_val = np.load('val_clusters_{}_{}_{}.npy'.format(model_name, model_number, n_clusters))
        cluster_ids_y = np.load('test_clusters_{}_{}_{}.npy'.format(model_name, model_number, n_clusters))

        train_bins = calculate_cluster_bins(cluster_ids_x, train_labels, n_clusters, n_classes)
        # val_bins = calculate_cluster_bins(cluster_ids_val, val_labels, n_clusters, n_classes)
        test_bins = calculate_cluster_bins(cluster_ids_y, test_labels, n_clusters, n_classes)
        #
        # equal_clusters = calculate_equal_clusters(train_bins)
        # #
        # cluster_ids_x = compress_clusters(cluster_ids_x, equal_clusters)
        # # cluster_ids_val = compress_clusters(cluster_ids_val, equal_clusters)
        # cluster_ids_y = compress_clusters(cluster_ids_y, equal_clusters)
        # # #
        # np.save('train_clusters_{}_{}_{}_compressed.npy'.format(model_name, model_number, n_clusters), cluster_ids_x)
        # # np.save('val_clusters_{}_{}_{}_compressed.npy'.format(model_name, model_number, n_clusters), cluster_ids_val)
        # np.save('test_clusters_{}_{}_{}_compressed.npy'.format(model_name, model_number, n_clusters), cluster_ids_y)
        # #
        # n_clusters = cluster_ids_x.max() + 1
        #
        # train_bins = calculate_cluster_bins(cluster_ids_x, train_labels, n_clusters, n_classes)
        # # val_bins = calculate_cluster_bins(cluster_ids_val, val_labels, n_clusters, n_classes)
        # test_bins = calculate_cluster_bins(cluster_ids_y, test_labels, n_clusters, n_classes)

        print_cluster_ids(train_bins)
        print("________________")
        # print_cluster_ids(val_bins)
        # print("________________")
        print_cluster_ids(test_bins)

        error_counter = 0
        error_pos = []
        n_zeros = 0
        for i in range(train_bins.shape[0]):
            tr_bin = train_bins[i]
            te_bin = test_bins[i]

            for j in range(tr_bin.shape[0]):
                if tr_bin[j] == 0 and te_bin[j] > 0:
                    error_counter += te_bin[j]
                    error_pos.append([i, j])
                if tr_bin[j] == 0:
                    n_zeros += 1
        print("Errors: {}".format(error_counter))
        print("Number zeros: {}".format(n_zeros))
        for pos in error_pos:
            print("Error bin: {} class: {}".format(pos[0], pos[1]))

        print('Sub-class problem: ')
        print(np.around((((n_clusters * n_classes) - n_zeros)/n_clusters), 2))

        # np.save('train_bins_{}_{}_{}'.format(model_name, model_number, n_clusters), train_bins)
        # np.save('test_bins_{}_{}_{}'.format(model_name, model_number, n_clusters), test_bins)
