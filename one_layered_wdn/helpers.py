import torch
import numpy as np
from sklearn import preprocessing

import itertools
import operator

# Constants
MNIST_DATASET = "MNIST"
FASHIONMNIST_DATASET = "Fashion_MNIST"
CIFAR10_DATASET = "CIFAR-10"

RELU_ACTIVATION = "RELU"
SELU_ACTIVATION = "SELU"

level_act_counter = None
used_models = []

def most_common(L):
    # get an iterable of (item, iterable) pairs
    SL = sorted((x, i) for i, x in enumerate(L))
    # print 'SL:', SL
    groups = itertools.groupby(SL, key=operator.itemgetter(0))

    # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        # print 'item %r, count %r, minind %r' % (item, count, min_index)
        return count, -min_index

    # pick the highest-count/earliest item
    return max(groups, key=_auxfun)[0]

def calculate_latent_vector(model, node, data, depth, latent_vector):
    global level_act_counter
    global used_models
    if depth == 0:
        values, familiar = model.is_familiar(node, data, provide_value=True)
        values = values.cpu().detach().numpy()
        latent_vector.append(values)
    lower_level_regions = model.divide_data_in_five(data)
    for region in lower_level_regions:
        for child_node in node.child_networks:
            if depth > 1:
                familiar, encoded_region = model.is_familiar(child_node, region, provide_encoding=True)
                calculate_latent_vector(model, child_node, encoded_region, depth - 1, latent_vector)
            else:
                calculate_latent_vector(model, child_node, region, depth - 1, latent_vector)


def convert_images_to_latent_vector(images, model):
    # global level_act_counter
    # global used_models
    # level_act_counter = np.zeros(model.n_levels)
    classifier_training_batch_size = min(images.data.shape[0], 1000)
    # classifier_training_batch_size = 1
    data_loader = torch.utils.data.DataLoader(images, batch_size=classifier_training_batch_size, shuffle=False)
    counter = 0
    features = []
    labels = []
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(model.device)
        # print(target)
        latent_vector = []
        for node in model.models:
            familiar, encoded_data = model.is_familiar(node, data, provide_encoding=True)
            calculate_latent_vector(model, node, encoded_data, model.n_levels - 1, latent_vector)
        # print(level_act_counter)
        # return None
        latent_vector = np.array(latent_vector)
        target_labels = target.cpu().detach().numpy()
        for i in range(classifier_training_batch_size):
            features.append(latent_vector[:, i])
            labels.append(target_labels[i])
            # labels.append(target_labels[i] == 5)
        counter += 1
        if counter % 10 == 0:
            print("Latent conversion iteration: ", counter)
        # break
    features = np.array(features)
    labels = np.array(labels, dtype=float)

    return features, None, labels


# def retrieve_expected_area(model, node, data, depth):
#     if depth == 0:
#         values, familiar = model.is_familiar(node, data, provide_value=True)
#     lower_level_regions = model.divide_data_in_five(data)
#     for region in lower_level_regions:
#         is_familiar = 0
#         for child_node in node.child_networks:
#             is_familiar = model.is_familiar(child_node, region)
#             if is_familiar == 1:
#                 break
#
#
# def convert_image_to_expected_image(image, network):
#     for m in network.models:
#





