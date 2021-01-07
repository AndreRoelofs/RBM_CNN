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
    print(train_features.shape)

    accuracies = []

    for cluster_id in range(n_clusters):
    # for cluster_id in range(37, 38):
        print("Current cluster ", cluster_id)
        train_cluster_idx = []
        for i in range(len(train_predictions)):
            cluster = train_predictions[i]
            if cluster != cluster_id:
                continue
            train_cluster_idx.append(i)

        train_dataset = UnsupervisedVectorDataset(train_features[train_cluster_idx], train_labels[train_cluster_idx])

        cluster_cnn_train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=min(10, len(train_predictions)),
            shuffle=False,
        )
        test_cluster_idx = []
        for i in range(len(test_predictions)):
            cluster = test_predictions[i]
            if cluster != cluster_id:
                continue
            test_cluster_idx.append(i)

        test_dataset = UnsupervisedVectorDataset(test_features[test_cluster_idx], test_labels[test_cluster_idx])

        cluster_cnn_test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=10,
            shuffle=False,
        )

        custom_svm = svm.Net(train_features.shape[1], 10)
        custom_svm.cuda()
        # custom_svm_optimizer = torch.optim.ASGD(custom_svm.parameters(), lr=1e-1)
        custom_svm_optimizer = torch.optim.Adam(custom_svm.parameters(), lr=1e-3)
        custom_svm_loss = svm.multiClassHingeLoss()
        best_epoch_idx = -1
        best_accuracy = 0.
        best_f1 = 0.
        history = list()
        for i in range(100):
            svm.train(i, custom_svm, custom_svm_optimizer, custom_svm_loss, cluster_cnn_train_dataloader)
            conf_mat, precision, recall, f1, accuracy = svm.test(i, custom_svm, cluster_cnn_test_dataloader,
                                                                 test_labels[test_cluster_idx])
            history.append((conf_mat, precision, recall, f1, accuracy))
            # if f1 > best_f1:
            if accuracy > best_accuracy:
                best_f1 = f1
                best_accuracy = accuracy
                best_epoch_idx = i
                # torch.save(custom_svm.state_dict(), 'best.model')

        print('Best epoch:{}\n'.format(best_epoch_idx))
        conf_mat, precision, recall, f1, accuracy = history[best_epoch_idx]
        print('conf_mat:\n', conf_mat)
        print('Precison:{:.4f}\nRecall:{:.4f}\nf1:{:.4f}\nAccuracy:{:.4f}'.format(precision, recall, f1, accuracy))
        accuracies.append(accuracy)

    print("Average accuracy over {} clusters is {}".format(n_clusters, np.mean(accuracies)))
    print(accuracies)


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    # wandb.init(project="wdn-v1")

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

        'min_familiarity_threshold': min_familiarity_threshold,

        'log_interval': 50

    }

    model = WDN(node_settings)

    for i in range(10):
        # for i in [5]:
        print("Training digit: ", i)
        subset_indices = (torch.tensor(train_data.targets) == i).nonzero().view(-1)
        model.train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=node_train_batch_size,
            shuffle=False,
            sampler=SubsetRandomSampler(subset_indices)
        )
        model.joint_training()

    # exit(0)

    print("Converting train images to latent vectors")
    train_features, _, train_labels = convert_images_to_latent_vector(train_data, model)
    print("Converting test images to latent vectors")
    test_features, _, test_labels = convert_images_to_latent_vector(test_data, model)
    print("Creating dataset of images")

    device = torch.device('cuda:0')
    # device = torch.device('cpu')
    n_clusters = 40
    print("Fitting clusters")

    train_features_tensor = torch.tensor(train_features, dtype=torch.float)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.int8)

    test_features_tensor = torch.tensor(test_features, dtype=torch.float)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.int8)

    cluster_ids_x, cluster_centers = kmeans(
        X=train_features_tensor, num_clusters=n_clusters, distance='euclidean', device=device
    )
    print("Predicting clusters")
    cluster_ids_y = kmeans_predict(
        test_features_tensor, cluster_centers, 'euclidean', device=device
    )

    calculate_average_accuracy_over_clusters(cluster_ids_x, cluster_ids_y, n_clusters)

    # train_dataset = UnsupervisedVectorDataset(train_features, train_labels)
    # train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=False)
    # #
    # test_dataset = UnsupervisedVectorDataset(test_features, test_labels)
    # test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)
    # #
    # print("Training classifier")
    # # fcnc = FullyConnectedClassifier(train_features.shape[1])
    # # fcnc_optimizer = torch.optim.Adam(fcnc.parameters(), lr=1e-3, amsgrad=True)
    # # #
    # # train_classifier(fcnc, fcnc_optimizer, train_dataset_loader, test_dataset_loader)
    # kmeans_train_features = train_features
    # kmeans_train_labels = train_labels
    #
    # kmeans_test_features = test_features
    # kmeans_test_labels = test_labels
    #
    # n_clusters = 40
    # kmeans = KMeans(n_clusters=n_clusters, random_state=0, max_iter=500, algorithm='elkan', n_jobs=-1).fit(
    #     train_features)
    # cluster_labels = kmeans.labels_
    # train_predictions = kmeans.predict(train_features)
    # test_predictions = kmeans.predict(test_features)
    #
    # calculate_average_accuracy_over_clusters(train_predictions, test_predictions, n_clusters)
    #
    # bins = np.zeros((n_clusters, 10))
    # for i in range(len(train_predictions)):
    #     cluster = train_predictions[i]
    #     bins[cluster][int(kmeans_train_labels[i])] += 1
    # bin_counter = 0
    # for bin in bins:
    #     print(bin_counter, np.array(bin, dtype=np.int))
    #     bin_counter += 1
    # print("_____________________")
    # #
    # test_bins = np.zeros((n_clusters, 10))
    # for i in range(len(cluster_ids_y)):
    #     cluster = cluster_ids_y[i]
    #     test_bins[cluster][int(test_labels[i])] += 1
    # bin_counter = 0
    # for bin in test_bins:
    #     print(bin_counter, np.array(bin, dtype=np.int))
    #     bin_counter += 1
    # np.save("20 clusters training bins", np.array(bins))
    # np.save("20 clusters test bins", np.array(test_bins))
    # #
    # # predictions = kmeans.predict(test_features)
    #
    # custom_svm = svm.Net(train_features.shape[1], 10)
    # custom_svm.cuda()
    # custom_svm_optimizer = torch.optim.SGD(custom_svm.parameters(), lr=1e-1)
    # custom_svm_loss = svm.multiClassHingeLoss()
    # best_epoch_idx = -1
    # best_f1 = 0.
    # history = list()
    # for i in range(90):
    #     svm.train(i, custom_svm, custom_svm_optimizer, custom_svm_loss, train_dataset_loader)
    #     conf_mat, precision, recall, f1 = svm.test(i, custom_svm, test_dataset_loader, test_labels)
    #     history.append((conf_mat, precision, recall, f1))
    #     if f1 > best_f1:  # save best model
    #         best_f1 = f1
    #         best_epoch_idx = i
    #         # torch.save(custom_svm.state_dict(), 'best.model')
    #
    # print('Best epoch:{}\n'.format(best_epoch_idx))
    # conf_mat, precision, recall, f1 = history[best_epoch_idx]
    # print('conf_mat:\n', conf_mat)
    # print('Precison:{:.4f}\nRecall:{:.4f}\nf1:{:.4f}\n'.format(precision, recall, f1))
    #
    # #
    # # #
    # svc = LinearSVC(max_iter=100, loss='hinge', random_state=0)
    # print("Fitting SVM")
    # # svc = SVC(cache_size=32768)
    # svc.fit(train_features, train_labels)
    # print("Predicting SVM")
    # predictions = svc.predict(test_features)
    # print('Result: %d/%d' % (sum(predictions == test_labels), test_labels.shape[0]))
    # #
    # wrong_indices = np.where(predictions != test_labels)[0]
    #
    # for i in wrong_indices:
    #     img = test_data.data[i].cpu().detach().numpy()
    #     plt.imshow(img, cmap='gray')
    #     plt.show()
