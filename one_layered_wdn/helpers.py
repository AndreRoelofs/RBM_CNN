import torch
import numpy as np
from sklearn import preprocessing

# Constants
MNIST_DATASET = "MNIST"
FASHIONMNIST_DATASET = "Fashion_MNIST"
CIFAR10_DATASET = "CIFAR-10"

RELU_ACTIVATION = "RELU"
SELU_ACTIVATION = "SELU"



def convert_images_to_latent_vector(images, model):
    classifier_training_batch_size = 1000
    classifier_training_batch_size = min(classifier_training_batch_size, images.data.shape[0])
    data_loader = torch.utils.data.DataLoader(images, batch_size=classifier_training_batch_size, shuffle=False)
    counter = 0
    features = []
    labels = []
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(model.device)
        latent_vector = []
        for m in model.models:
            values = model.is_familiar(m, data, provide_value=True)
            values = values.cpu().detach().numpy()
            latent_vector.append(values)
            second_level_regions = model.divide_data_in_four(data)
            for second_level_region in second_level_regions:
                for m_second_level in m.child_networks:
                    values = model.is_familiar(m_second_level, second_level_region, provide_value=True)
                    values = values.cpu().detach().numpy()
                    latent_vector.append(values)
                    third_level_regions = model.divide_data_in_four(second_level_region)
                    for third_level_region in third_level_regions:
                        for m_third_level in m_second_level.child_networks:
                            values = model.is_familiar(m_third_level, third_level_region, provide_value=True)
                            values = values.cpu().detach().numpy()
                            latent_vector.append(values)
                            fourth_level_regions = model.divide_data_in_four(third_level_region)
                            for fourth_level_region in fourth_level_regions:
                                for m_fourth_level in m_third_level.child_networks:
                                    values = model.is_familiar(m_fourth_level, fourth_level_region, provide_value=True)
                                    values = values.cpu().detach().numpy()
                                    latent_vector.append(values)
                                    fifth_level_regions = model.divide_data_in_four(fourth_level_region)
                                    for fifth_level_region in fifth_level_regions:
                                        for m_fifth_level in m_fourth_level.child_networks:
                                            values = model.is_familiar(m_fifth_level, fifth_level_region, provide_value=True)
                                            values = values.cpu().detach().numpy()
                                            latent_vector.append(values)
        latent_vector = np.array(latent_vector)
        target_labels = target.cpu().detach().numpy()
        for i in range(classifier_training_batch_size):
            features.append(latent_vector[:, i])
            # labels.append(target_labels[i] == 5)
            labels.append(target_labels[i])
        counter += 1
        if counter % 10 == 0:
            print("Latent conversion iteration: ", counter)
    # new_dataset = np.array(new_dataset, dtype=float)
    features = np.array(features)

    features_norm = preprocessing.scale(features)
    labels = np.array(labels, dtype=float)

    return features, features_norm, labels
