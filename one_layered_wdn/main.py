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
from one_layered_wdn.wdn import WDN, train_wdn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from one_layered_wdn.custom_dataset import UnsupervisedVectorDataset
from one_layered_wdn.custom_classifier import FullyConnectedClassifier, train_classifier, predict_classifier, FashionCNN
import one_layered_wdn.svm as svm
from torch import nn
from kmeans_pytorch import kmeans, kmeans_predict
import copy

# import wandb


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
                                      transforms.Normalize((0.1307,), (0.3081,)),
                                      # CropBlackPixelsAndResize(tol=tolerance, output_size=image_input_size),
                                      # transforms.Resize((image_input_size, image_input_size)),
                                  ]))

        test_data = FashionMNIST(data_path, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            # CropBlackPixelsAndResize(tol=tolerance, output_size=image_input_size),
            # transforms.Resize((image_input_size, image_input_size)),
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


def calculate_average_accuracy_over_clusters(train_predictions, test_predictions, n_clusters):
    np.random.seed(0)
    torch.manual_seed(0)

    global big_cnnc

    accuracies = []
    if big_cnnc is None:
    # if True:
        cluster_cnn_train_dataloader = torch.utils.data.DataLoader(train_data,
                                                                   batch_size=100,
                                                                   shuffle=True,
                                                                   )
        cluster_cnn_test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True,
                                                                  )
        big_cnnc = FashionCNN()
        cnnc_optimizer = torch.optim.Adam(big_cnnc.parameters(), lr=1e-3, amsgrad=False)
        train_classifier(big_cnnc, cnnc_optimizer, cluster_cnn_train_dataloader, cluster_cnn_test_dataloader, [], 5)
        for param in big_cnnc.parameters():
            param.requires_grad = False
        del cluster_cnn_train_dataloader
        del cluster_cnn_test_dataloader

    # for cluster_id in range(n_clusters):
    for cluster_id in range(0, 1):
        print("Current cluster ", cluster_id)
        train_cluster_idx = []
        for i in range(len(train_predictions)):
            cluster = train_predictions[i]
            if cluster != cluster_id:
                continue
            train_cluster_idx.append(i)


        # for idx in train_cluster_idx[58:59]:
        #     plt.imshow(train_data.data[idx].reshape((28, 28)).cpu().detach().numpy(), cmap='gray')
        #     plt.show()
        #
        # return


        cluster_cnn_train_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=min(10, len(train_cluster_idx)),
            # batch_size=len(train_cluster_idx),
            shuffle=False,
            sampler=SubsetRandomSampler(train_cluster_idx)
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

        cnnc_optimizer = torch.optim.Adam(big_cnnc_clone.parameters(), lr=1e-4, amsgrad=False)
        # cnnc_optimizer = torch.optim.SGD(big_cnnc_clone.parameters(), lr=1e-3)
        train_classifier(big_cnnc_clone, cnnc_optimizer, cluster_cnn_train_dataloader, cluster_cnn_test_dataloader,
                         accuracies, 5)
        print("General Classifier:")
        predict_classifier(big_cnnc, cluster_cnn_test_dataloader, [])

        big_cnnc_clone.cpu()
        del big_cnnc_clone
        del cluster_cnn_train_dataloader
        del cluster_cnn_test_dataloader

    print("Average accuracy over {} clusters is {}".format(n_clusters, np.sum(accuracies)))
    print(accuracies)


def train_knn(train_features, test_features, n_clusters):
    device = torch.device('cuda:0')
    # device = torch.device('cpu')
    tr_features = torch.tensor(train_features, dtype=torch.float)

    # tr_features -= tr_features.min(0, keepdim=True)[0]
    # tr_features /= tr_features.max(0, keepdim=True)[0]

    te_features = torch.tensor(test_features, dtype=torch.float)
    #
    # te_features -= te_features.min(0, keepdim=True)[0]
    # te_features /= te_features.max(0, keepdim=True)[0]

    cluster_ids_x, cluster_centers = kmeans(
        X=tr_features, num_clusters=n_clusters,
        distance='euclidean',
        # distance='cosine',
        device=device
    )

    cluster_ids_y = kmeans_predict(te_features, cluster_centers,
                                   distance='euclidean',
                                   # distance='cosine',
                                   device=device)

    # kmeans = KMeans(n_clusters=n_clusters, random_state=0, max_iter=100, algorithm='elkan', n_jobs=-1).fit(
    #     train_features)
    # cluster_ids_x = kmeans.predict(train_features)
    # cluster_ids_y = kmeans.predict(test_features)

    return cluster_ids_x, cluster_ids_y


def print_cluster_ids(cluster_ids, data_labels, n_clusters=10):
    bins = np.zeros((n_clusters, 10))
    for i in range(len(cluster_ids)):
        cluster = cluster_ids[i]
        bins[cluster][int(data_labels[i])] += 1
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
    return bins

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    # wandb.init(project="wdn-v1")

    # config = wandb.config

    config = configparser.ConfigParser()
    config.read('config.ini')

    process_settings()

    load_data()

    wdn_settings = {
        'image_input_size': image_input_size,
        'image_channels': input_filters,

        'encoder_channels': encoder_output_filters,
        'encoder_learning_rate': encoder_learning_rate,
        'encoder_activation': encoder_activation,

        'rbm_visible_units': rbm_visible_units,
        'rbm_hidden_units': rbm_hidden_units,
        'rbm_learning_rate': rbm_learning_rate,
        'rbm_activation': rbm_activation,

        'min_familiarity_threshold': min_familiarity_threshold,

        'log_interval': 50

    }
    print("Train WDN")
    model = train_wdn(train_data, wdn_settings)

    #old_train_features = np.load('old_train_features.npy')
    print("Convert train images to latent vectors")
    train_features, _, train_labels = convert_images_to_latent_vector(train_data, model)
    # train_features, _, train_labels = old_convert_images_to_latent_vector(train_data, model)
    print("Convert test images to latent vectors")
    test_features, _, test_labels = convert_images_to_latent_vector(test_data, model)
    # device = torch.device('cuda:0')
    # train_features = torch.tensor(train_features, dtype=torch.float, device=device)
    # train_labels = torch.tensor(train_labels, dtype=torch.int, device=device)
    #
    # test_features = torch.tensor(test_features, dtype=torch.float, device=device)
    # test_labels = torch.tensor(test_labels, dtype=torch.int, device=device)

    # print("Fitting SVM")
    # # svc = LinearSVC(max_iter=10000, loss='hinge', random_state=0)
    # svc = SVC(cache_size=32768, tol=1e-3, kernel='linear', random_state=0)
    # svc.fit(train_features, train_labels)
    # print("Predicting SVM")
    # predictions = svc.predict(train_features)
    # print('Train Result: %d/%d' % (np.sum(predictions == train_labels), train_labels.shape[0]))
    # predictions = svc.predict(test_features)
    # print('Test Result: %d/%d' % (np.sum(predictions == test_labels), test_labels.shape[0]))
    #
    # exit(1)
    #
    # train_features = np.load('3_level_train_features.npy')
    # train_labels = np.load('3_level_train_labels.npy')
    # test_features = np.load('3_level_test_features.npy')
    # test_labels = np.load('3_level_test_labels.npy')
    #
    n_clusters = 40
    print("Fit KNN")
    cluster_ids_x, cluster_ids_y = train_knn(train_features, test_features, n_clusters)
    #
    # np.save("3_level_train_clusters_40_cosine.npy", cluster_ids_x)
    # np.save("3_level_test_clusters_40_cosine.npy", cluster_ids_y)

    # exit(1)
    #
    # cluster_ids_x = np.load("3_level_train_clusters_10_cosine.npy")
    # cluster_ids_y = np.load("3_level_test_clusters_10_cosine.npy")
    # print("Train predictor")
    # calculate_average_accuracy_over_clusters(cluster_ids_x, cluster_ids_y, n_clusters)
    #
    # print("Creating dataset of images")
    # train_dataset = UnsupervisedVectorDataset(train_features, train_labels)
    # train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=False)
    # #
    # test_dataset = UnsupervisedVectorDataset(test_features, test_labels)
    # test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)
    #
    # print("Training classifier")
    # fcnc = FullyConnectedClassifier(train_features.shape[1], 10)
    # fcnc_optimizer = torch.optim.Adam(fcnc.parameters(), lr=1e-3, amsgrad=False)
    # #
    # train_classifier(fcnc, fcnc_optimizer, train_dataset_loader, test_dataset_loader, [])

    train_bins = print_cluster_ids(cluster_ids_x, train_labels, n_clusters=n_clusters)
    print("_______________-")
    test_bins = print_cluster_ids(cluster_ids_y, test_labels, n_clusters=n_clusters)

    error_counter = 0
    for i in range(train_bins.shape[0]):
        tr_bin = train_bins[i]
        te_bin = test_bins[i]

        for j in range(tr_bin.shape[0]):
            if tr_bin[j] == 0 and te_bin[j] > 0:
                error_counter += te_bin[j]
    print("Errors: {}".format(error_counter))

    # test_bins = np.zeros((n_clusters, 10))
    # for i in range(len(cluster_ids_y)):
    #     cluster = cluster_ids_y[i]
    #     test_bins[cluster][int(test_labels[i])] += 1
    # bin_counter = 0
    # for bin in test_bins:
    #     print(bin_counter, np.array(bin, dtype=np.int))
    #     bin_counter += 1

    # np.save("3_level_train_features.npy", train_features)
    # np.save("3_level_train_labels.npy", train_labels)
    # np.save("3_level_test_features.npy", test_features)
    # np.save("3_level_test_labels.npy", test_labels)

    # svc = LinearSVC(max_iter=100, loss='hinge', random_state=0)
    # print("Fitting SVM")
    # # svc = SVC(cache_size=32768)
    # svc.fit(train_features, train_labels)
    # print("Predicting SVM")
    # predictions = svc.predict(train_features)
    # print('Train Result: %d/%d' % (np.sum(predictions == train_labels), train_labels.shape[0]))
    # predictions = svc.predict(test_features)
    # print('Test Result: %d/%d' % (np.sum(predictions == test_labels), test_labels.shape[0]))
    # #
    # wrong_indices = np.where(predictions != test_labels)[0]

    # for i in wrong_indices:
    #     img = test_data.data[i].cpu().detach().numpy()
    #     plt.imshow(img, cmap='gray')
    #     plt.show()
