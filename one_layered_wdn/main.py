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
from kmeans_pytorch import kmeans, kmeans_predict
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
                                      # CropBlackPixelsAndResize(tol=tolerance, output_size=image_input_size),
                                      transforms.Resize((image_input_size, image_input_size)),
                                  ]))

        test_data = FashionMNIST(data_path, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            # CropBlackPixelsAndResize(tol=tolerance, output_size=image_input_size),
            transforms.Resize((image_input_size, image_input_size)),
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

    accuracies = []
    low_performance_clusters = [0, 1, 5, 11, 14, 18, 21, 22, 25, 28, 30, 31, 36]
    low_performance_clusters = []
    if len(low_performance_clusters) > 0:
        cluster_cnn_train_dataloader = torch.utils.data.DataLoader(train_data,
                                                                   batch_size=100,
                                                                   shuffle=True,
                                                                   )
        cluster_cnn_test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False,
                                                                  )
        big_cnnc = FashionCNN()
        cnnc_optimizer = torch.optim.Adam(big_cnnc.parameters(), lr=1e-3, amsgrad=False)
        train_classifier(big_cnnc, cnnc_optimizer, cluster_cnn_train_dataloader, cluster_cnn_test_dataloader, [])

    # for cluster_id in range(n_clusters):
    for cluster_id in range(1, 2):
        print("Current cluster ", cluster_id)
        train_cluster_idx = []
        for i in range(len(train_predictions)):
            cluster = train_predictions[i]
            if cluster != cluster_id:
                continue
            train_cluster_idx.append(i)

        cluster_cnn_train_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=min(10, len(train_cluster_idx)),
            shuffle=False,
            sampler=SubsetRandomSampler(train_cluster_idx)
        )
        test_cluster_idx = []
        for i in range(len(test_predictions)):
            cluster = test_predictions[i]
            if cluster != cluster_id:
                continue
            test_cluster_idx.append(i)

        # test_dataset = UnsupervisedVectorDataset(test_features[test_cluster_idx], test_labels[test_cluster_idx])

        cluster_cnn_test_dataloader = torch.utils.data.DataLoader(
            test_data,
            batch_size=100,
            shuffle=False,
            sampler=SubsetRandomSampler(test_cluster_idx)
        )

        if cluster_id in low_performance_clusters:
            predict_classifier(big_cnnc, cluster_cnn_test_dataloader, accuracies)
        else:
            cnnc = FashionCNN()
            cnnc_optimizer = torch.optim.Adam(cnnc.parameters(), lr=1e-3, amsgrad=False)
            train_classifier(cnnc, cnnc_optimizer, cluster_cnn_train_dataloader, cluster_cnn_test_dataloader,
                             accuracies)

    print("Average accuracy over {} clusters is {}".format(n_clusters, np.average(accuracies)))
    print(accuracies)

def train_knn(train_features, test_features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, max_iter=100, algorithm='elkan', n_jobs=-1).fit(
        train_features)
    cluster_ids_x = kmeans.predict(train_features)
    cluster_ids_y = kmeans.predict(test_features)

    return cluster_ids_x, cluster_ids_y



if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    # wandb.init(project="wdn-v1")

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

    print("Convert train images to latent vectors")
    train_features, _, train_labels = convert_images_to_latent_vector(train_data, model)
    print("Convert test images to latent vectors")
    test_features, _, test_labels = convert_images_to_latent_vector(test_data, model)

    print("Fitting SVM")
    # svc = LinearSVC(max_iter=10000, loss='hinge', random_state=0)
    svc = SVC(cache_size=32768, tol=1e-10, kernel='linear')
    svc.fit(train_features, train_labels)
    print("Predicting SVM")
    predictions = svc.predict(train_features)
    print('Train Result: %d/%d' % (np.sum(predictions == train_labels), train_labels.shape[0]))
    predictions = svc.predict(test_features)
    print('Test Result: %d/%d' % (np.sum(predictions == test_labels), test_labels.shape[0]))


    #
    # # train_features = np.load('3_level_train_features.npy')
    # # train_labels = np.load('3_level_train_labels.npy')
    # # test_features = np.load('3_level_test_features.npy')
    # # test_labels = np.load('3_level_test_labels.npy')
    # # cluster_ids_x = np.load("3_level_train_clusters.npy")
    # # cluster_ids_y = np.load("3_level_test_clusters.npy")
    #
    # n_clusters = 40
    # print("Fit KNN")
    # cluster_ids_x, cluster_ids_y = train_knn(train_features, test_features, n_clusters)
    #
    #
    # print("Train predictor")
    # calculate_average_accuracy_over_clusters(cluster_ids_x, cluster_ids_y, n_clusters)
    #
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

