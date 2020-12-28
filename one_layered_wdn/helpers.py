import torch
import numpy as np
from sklearn import preprocessing

# Constants
MNIST_DATASET = "MNIST"
FASHIONMNIST_DATASET = "Fashion_MNIST"
CIFAR10_DATASET = "CIFAR-10"

RELU_ACTIVATION = "RELU"
SELU_ACTIVATION = "SELU"


# def convert_images_to_latent_vector(images, model):
#     classifier_training_batch_size = 1
#     data_loader = torch.utils.data.DataLoader(images, batch_size=classifier_training_batch_size, shuffle=False)
#     counter = 0
#     features = []
#     labels = []
#     failure_value = 10.0
#     for batch_idx, (data, target) in enumerate(data_loader):
#         data = data.to(model.device)
#         latent_vector = []
#         for m in model.models:
#             # Encode the image
#             rbm_input = m.encode(data)
#             flat_rbm_input = rbm_input.view(len(rbm_input), model.model_settings['rbm_visible_units'])
#
#             familiar = m.rbm.is_familiar(flat_rbm_input, provide_value=False)
#
#             if familiar == 1:
#                 second_level_regions = model.generate_second_level_regions(data)
#                 for second_level_region in second_level_regions:
#                     for m_second_level in m.child_networks:
#                         second_level_familiar = model.is_familiar(m_second_level, second_level_region)
#                         if second_level_familiar == 1:
#                             third_level_regions = model.generate_third_level_regions(second_level_region)
#                             for third_level_region in third_level_regions:
#                                 for m_third_level in m_second_level.child_networks:
#                                     value, third_level_familiar = model.is_familiar(m_third_level, third_level_region,
#                                                                                     provide_value=True)
#
#                                     if third_level_familiar == 1:
#                                         latent_vector.append(value.cpu().detach().numpy()[0])
#                                     else:
#                                         latent_vector.append(failure_value)
#                         else:
#                             for i in range(len(m_second_level.child_networks)):
#                                 for j in range(4):
#                                     latent_vector.append(failure_value)
#             else:
#                 for m_second_level in m.child_networks:
#                     for z in range(4):
#                         for i in range(len(m_second_level.child_networks)):
#                             for j in range(4):
#                                 latent_vector.append(failure_value)
#
#         target_labels = target.cpu().detach().numpy()
#
#         features.append(np.array(latent_vector))
#         #labels.append(target_labels[0] == 1)
#         labels.append(target_labels[0])
#
#         counter += 1
#         if counter % 10 == 0:
#             print("Latent conversion iteration: ", counter)
#     # new_dataset = np.array(new_dataset, dtype=float)
#     features = np.array(features)
#
#     features_norm = preprocessing.scale(features)
#     labels = np.array(labels, dtype=float)
#
#     return features, features_norm, labels

def convert_images_to_latent_vector(images, model):
    classifier_training_batch_size = 100
    data_loader = torch.utils.data.DataLoader(images, batch_size=classifier_training_batch_size, shuffle=False)
    counter = 0
    features = []
    labels = []
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(model.device)
        latent_vector = []
        for m in model.models:
            values, _ = model.is_familiar(m, data, provide_value=True)
            values = values.cpu().detach().numpy()
            latent_vector.append(values)
            second_level_regions = model.generate_second_level_regions(data)
            for second_level_region in second_level_regions:
                for m_second_level in m.child_networks:
                    values, _ = model.is_familiar(m_second_level, second_level_region, provide_value=True)
                    values = values.cpu().detach().numpy()
                    latent_vector.append(values)
                    third_level_regions = model.generate_third_level_regions(second_level_region)
                    for third_level_region in third_level_regions:
                        for m_third_level in m_second_level.child_networks:
                            values, _ = model.is_familiar(m_third_level, third_level_region, provide_value=True)
                            values = values.cpu().detach().numpy()
                            latent_vector.append(values)

        latent_vector = np.array(latent_vector)
        target_labels = target.cpu().detach().numpy()
        for i in range(classifier_training_batch_size):
            features.append(latent_vector[:, i])
            labels.append(target_labels[i] == 1)
            # labels.append(target_labels[i])
        counter += 1
        if counter % 100 == 0:
            print("Training iteration: ", counter)
    # new_dataset = np.array(new_dataset, dtype=float)
    features = np.array(features)

    features_norm = preprocessing.scale(features)
    labels = np.array(labels, dtype=float)

    return features, features_norm, labels
