import torch
import numpy as np
from sklearn import preprocessing

# Constants
MNIST_DATASET = "MNIST"
FASHIONMNIST_DATASET = "Fashion_MNIST"
CIFAR10_DATASET = "CIFAR-10"

RELU_ACTIVATION = "RELU"
SELU_ACTIVATION = "SELU"

def calculate_latent_vector(model, node, data, depth):
    latent_vector = []
    lower_level_regions = model.divide_data_in_five(data)
    for region in lower_level_regions:
        for child_node in node.child_networks:
            if depth-1 == 0:
                values, familiar = model.is_familiar(node, region)
                values = values.cpu().detach().numpy()
                latent_vector.append(values)
            else:
                calculate_latent_vector(model, child_node, region, depth - 1)
    return latent_vector


def convert_images_to_latent_vector(images, model):
    classifier_training_batch_size = images.data.shape[0]
    data_loader = torch.utils.data.DataLoader(images, batch_size=classifier_training_batch_size, shuffle=False)
    counter = 0
    features = []
    labels = []
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(model.device)
        latent_vector = []
        # correct_indices = []
        for node in model.models:
            latent_vector += calculate_latent_vector(model, node, data, model.n_levels)
        latent_vector = np.array(latent_vector)
        target_labels = target.cpu().detach().numpy()
        for i in range(classifier_training_batch_size):
            features.append(latent_vector[:, i])
            labels.append(target_labels[i])
        counter += 1
        if counter % 10 == 0:
            print("Latent conversion iteration: ", counter)
    # new_dataset = np.array(new_dataset, dtype=float)
    features = np.array(features)

    # features_norm = preprocessing.scale(features)
    labels = np.array(labels, dtype=float)

    return features, features, labels
