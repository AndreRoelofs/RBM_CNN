import torch
import numpy as np
from sklearn import preprocessing
import copy

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


def calculate_latent_vector(model, node, data, depth, latent_vector, latent_vector_id, parent_mask=None):
    if depth == 0:
        return model.is_familiar(node, data, provide_value=True)[0]

    lower_level_regions = model.divide_data_in_five(data)
    child_counter = 0
    for child_node in node.child_networks:
        max_values = None
        for region in lower_level_regions:
            child_values = calculate_latent_vector(
                model, child_node, region, depth - 1,
                latent_vector, latent_vector_id + child_counter)
            if max_values is None:
                max_values = child_values
            else:
                max_values = torch.where(max_values > child_values, max_values, child_values)

        if depth == 1:
            max_values = max_values.cpu().detach().numpy()
            latent_vector[:, latent_vector_id + child_counter] = max_values
        child_counter += 1
    return None


def convert_images_to_latent_vector(images, model):
    # global level_act_counter
    # global used_models
    # level_act_counter = np.zeros(model.n_levels)
    n_data = images.data.shape[0]
    batch_size = min(1000, len(images))
    # classifier_training_batch_size = 1
    data_loader = torch.utils.data.DataLoader(images, batch_size=batch_size, shuffle=False)
    counter = 0
    features = np.zeros((n_data, model.models_total))
    labels = np.zeros(n_data)
    # features = []
    for batch_idx, (data, target) in enumerate(data_loader):
        # counter += 1
        # if counter < 4:
        #     continue
        data = data.to(model.device)
        latent_vector = np.zeros((batch_size, model.models_total))
        # latent_vector = []
        latent_vector_id = 0
        for node in model.models:
            encoded_data = node.encode(data)
            values = calculate_latent_vector(model, node, encoded_data, model.n_levels - 1, latent_vector,
                                             latent_vector_id)
            if model.n_levels == 1:
                latent_vector[:, latent_vector_id] = values.cpu().detach().numpy()
            # if not familiar:
            #     latent_vector += np.repeat(values, node.n_children).tolist()
            # unfamiliar_values = np.repeat(values, node.n_children)
            # for i in range(data.shape[0]):
            #     if familiar[i] == 0:
            #         update_latent_vector(node, values, latent_vector, latent_vector_id, familiar.eq(0), model.n_levels)
            latent_vector_id += node.n_children

        target_labels = target.cpu().detach().numpy()

        features[batch_idx * batch_size:batch_idx * batch_size + batch_size] = latent_vector
        labels[batch_idx * batch_size:batch_idx * batch_size + batch_size] = target_labels
        counter += 1
        if counter % 100 == 0:
            print("Latent conversion iteration: ", counter)
        # break

    return features, None, labels


def old_calculate_latent_vector(model, node, data, depth, latent_vector):
    global level_act_counter
    global used_models
    if depth == 0:
        values, familiar = model.is_familiar(node, data, provide_value=True)
        # values = values.cpu().detach().numpy()
        return values
    lower_level_regions = model.divide_data_in_five(data)
    for child_node in node.child_networks:
        max_values = None
        for region in lower_level_regions:
            if depth > 1:
                familiar, encoded_region = model.is_familiar(child_node, region, provide_encoding=True)
                old_calculate_latent_vector(model, child_node, encoded_region, depth - 1, latent_vector)
            else:
                values = old_calculate_latent_vector(model, child_node, region, depth - 1, latent_vector)
                if max_values is None:
                    max_values = values

                max_values = torch.where(values > max_values, values, max_values)
        if max_values is not None:
            latent_vector.append(max_values.cpu().detach().numpy())

    return None


def old_convert_images_to_latent_vector(images, model):
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
            if model.n_levels > 1:
                familiar, encoded_data = model.is_familiar(node, data, provide_encoding=True)
                old_calculate_latent_vector(model, node, encoded_data, model.n_levels - 1, latent_vector)
            else:
                values, familiar = model.is_familiar(node, data, provide_value=True)
                values = values.cpu().detach().numpy()
                latent_vector.append(values)

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
# #
