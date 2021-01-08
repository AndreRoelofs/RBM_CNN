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
    values, familiar = model.is_familiar(node, data, provide_value=True)
    values = values.cpu().detach().numpy()
    if familiar == 0:
        return False, values
    if depth == 0:
        latent_vector += np.repeat(values, node.n_children).tolist()
        return True, values
    lower_level_regions = model.divide_data_in_five(data)
    for child_node in node.child_networks:
        is_region_familiar = False
        for region in lower_level_regions:
            is_familiar, child_values = calculate_latent_vector(model, child_node, region, depth - 1, latent_vector)
            if is_familiar:
                is_region_familiar = True
                break
        if not is_region_familiar:
            # latent_vector += np.repeat(child_values, child_node.n_children).tolist()
            latent_vector += np.repeat(child_values, child_node.n_children).tolist()

    return True, values


def convert_images_to_latent_vector(images, model):
    # global level_act_counter
    # global used_models
    # level_act_counter = np.zeros(model.n_levels)
    n_data = images.data.shape[0]
    batch_size = 1
    # classifier_training_batch_size = 1
    data_loader = torch.utils.data.DataLoader(images, batch_size=batch_size, shuffle=False)
    counter = 0
    features = np.zeros((n_data, model.models_total))
    labels = np.zeros(n_data)
    # features = []
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(model.device)
        # latent_vector = np.zeros((batch_size, model.models_total))
        latent_vector = []
        for node in model.models:
            familiar, values = calculate_latent_vector(model, node, data, model.n_levels - 1, latent_vector)
            if not familiar:
                latent_vector += np.repeat(values, node.n_children).tolist()

        target_labels = target.cpu().detach().numpy()

        features[batch_idx * batch_size:batch_idx * batch_size + batch_size] = latent_vector
        labels[batch_idx * batch_size:batch_idx * batch_size + batch_size] = target_labels
        counter += 1
        if counter % 100 == 0:
            print("Latent conversion iteration: ", counter)
        # break

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





