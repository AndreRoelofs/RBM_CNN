import torch
import numpy as np
from sklearn import preprocessing
import copy
from torchvision.transforms.functional import hflip
import itertools
import operator
import random

# Constants
MNIST_DATASET = "MNIST"
FASHIONMNIST_DATASET = "Fashion_MNIST"
CIFAR10_DATASET = "CIFAR_10"
CIFAR100_DATASET = "CIFAR_100"

RELU_ACTIVATION = "RELU"
SELU_ACTIVATION = "SELU"

level_act_counter = None
used_models = []

fashion_mnist_classes = [
    'T-shirt',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot']

cifar10_classes = [
    'Airplane',
    'Automobile',
    'Bird',
    'Cat',
    'Deer',
    'Dog',
    'Frog',
    'Horse',
    'Ship',
    'Truck']



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

def reset_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


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
            latent_vector[:, latent_vector_id + child_counter] = np.where(
                latent_vector[:, latent_vector_id + child_counter] > max_values,
                latent_vector[:, latent_vector_id + child_counter], max_values)

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
        latent_vector = np.zeros((batch_size, model.models_total)) - 10000
        # latent_vector = []
        latent_vector_id = 0
        for node in model.models:
            # encoded_data = node.encode(data)
            if model.n_levels == 1:
                values = calculate_latent_vector(model, node, data, model.n_levels - 1, latent_vector,
                                                 latent_vector_id)
                latent_vector[:, latent_vector_id] = values
            else:
                calculate_latent_vector(model, node, data, model.n_levels - 1, latent_vector,
                                        latent_vector_id)
            latent_vector_id += node.n_children

        target_labels = target.cpu().detach().numpy()

        features[batch_idx * batch_size:batch_idx * batch_size + batch_size] = latent_vector
        labels[batch_idx * batch_size:batch_idx * batch_size + batch_size] = target_labels
        counter += 1
        if counter % 10 == 0:
            print("Latent conversion iteration: ", counter)
        # break

    return features, None, labels

# #
