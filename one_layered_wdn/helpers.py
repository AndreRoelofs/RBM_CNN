# Constants
import torch
import numpy as np
from sklearn import preprocessing

MNIST_DATASET = "MNIST"
FASHIONMNIST_DATASET = "Fashion_MNIST"
CIFAR10_DATASET = "CIFAR-10"

RELU_ACTIVATION = "RELU"
SELU_ACTIVATION = "SELU"


def convert_images_to_latent_vector(images, model):
    classifier_training_batch_size = 1000
    data_loader = torch.utils.data.DataLoader(images, batch_size=classifier_training_batch_size, shuffle=False)
    counter = 0
    features = []
    labels = []
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(model.device)
        latent_vector = []
        for m in model.models:
            # Encode the image
            rbm_input = m.encode(data)
            flat_rbm_input = rbm_input.view(len(rbm_input), model.model_settings['rbm_visible_units'])

            # Compare data with existing models
            values = m.rbm.is_familiar(flat_rbm_input)
            values = values.cpu().detach().numpy()
            latent_vector.append(values)

        latent_vector = np.array(latent_vector)
        target_labels = target.cpu().detach().numpy()
        for i in range(classifier_training_batch_size):
            # test_target = np.zeros(10, dtype=float)
            # test_target[target_labels[i]] = 1.0
            # training_features.append(latent_vector[:, i])
            features.append(latent_vector[:, i])
            labels.append(target_labels[i] == 1)
            # labels.append(test_target == 1)

        counter += 1
        if counter % 100 == 0:
            print("Training iteration: ", counter)
    # new_dataset = np.array(new_dataset, dtype=float)
    features = np.array(features)
    features_norm = preprocessing.scale(features)
    labels = np.array(labels, dtype=float)

    return features, features_norm, labels



