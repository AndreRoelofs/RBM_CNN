import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import resize, crop, rotate, affine, center_crop, five_crop, ten_crop, vflip, \
    hflip
import matplotlib.pyplot as plt
import numpy as np
from one_layered_wdn.node import Node
from one_layered_wdn.helpers import *


class WDN(nn.Module):
    def __init__(self, model_settings):
        super().__init__()
        self.device = torch.device("cuda")

        self.models = []
        self.model_settings = model_settings
        # self.create_new_model()

        self.levels = [
            # {'input_channels': 1, 'encoder_channels': 1, 'rbm_visible_units': 112,  'rbm_hidden_units': 800, 'rbm_learning_rate': 1e-20},
            # {'input_channels': 1, 'encoder_channels': 16, 'rbm_visible_units': 56, 'encoder_weight_variance': 0.5,
            #  'rbm_hidden_units': 100, 'rbm_learning_rate': 1e-4},
            {'input_channels': 1, 'encoder_channels': 4, 'rbm_visible_units': 28, 'encoder_weight_variance': 1.0,
             'rbm_hidden_units': 300, 'rbm_learning_rate': 1e-3},
            {'input_channels': 1, 'encoder_channels': 16, 'rbm_visible_units': 14, 'encoder_weight_variance': 2.0,
             'rbm_hidden_units': 5, 'rbm_learning_rate': 1e-5},
            {'input_channels': 1, 'encoder_channels': 64, 'rbm_visible_units': 7, 'encoder_weight_variance': 20.0,
             'rbm_hidden_units': 2, 'rbm_learning_rate': 1e-10},
            {'input_channels': 1, 'encoder_channels': 64, 'rbm_visible_units': 3, 'encoder_weight_variance': 8.0,
             'rbm_hidden_units': 2, 'rbm_learning_rate': 1e-10},
            {'input_channels': 1, 'encoder_channels': 1, 'rbm_visible_units': 2, 'encoder_weight_variance': 20.0,
             'rbm_hidden_units': 5, 'rbm_learning_rate': 1e-3},
        ]

        self.n_levels = 3

        self.log_interval = 100

    def create_new_model(self, level, target):

        settings = self.levels[level]

        network = Node(
            image_channels=settings['input_channels'],
            encoder_channels=settings['encoder_channels'],
            encoder_weight_variance=settings['encoder_weight_variance'],
            rbm_visible_units=(settings['rbm_visible_units'] ** 2) * settings['encoder_channels'],
            rbm_hidden_units=settings['rbm_hidden_units'],
            rbm_learning_rate=settings['rbm_learning_rate'],
            use_relu=self.model_settings['rbm_activation'] == RELU_ACTIVATION,
            level=level
        )
        network.to(self.device)

        return network

    def resize_image(self, image):
        return resize(image, [self.model_settings['image_input_size'], self.model_settings['image_input_size']])

    def loss_function(self, recon_x, x):
        return F.mse_loss(x, recon_x)

    def divide_data_in_five(self, data):
        original_size = data.shape[-1]
        # Accepts images of up to 128x128 size, more needed?
        new_size = np.floor(original_size / 2).astype(np.int64)
        new_size = max(new_size, 2)

        # offset = new_size
        #
        # while offset + new_size > original_size:
        #     offset -= 1
        #
        # regions = [
        #     crop(data, 0, 0, new_size, new_size),
        #     crop(data, 0, offset, new_size, new_size),
        #     crop(data, offset, 0, new_size, new_size),
        #     crop(data, offset, offset, new_size, new_size),
        #     center_crop(data, [new_size, new_size]),
        # ]

        # cropped_regions = five_crop(data, [new_size, new_size])
        # regions = []
        # for i in range(5):
        #     regions.append(cropped_regions[i])
        #     regions.append(hflip(cropped_regions[i]))
        #     regions.append(vflip(cropped_regions[i]))
        #
        # return regions
        return five_crop(data, [new_size, new_size])
        # return ten_crop(data, [new_size, new_size])

        # return regions

    def is_familiar(self, network, data, provide_value=False):
        # Encode the image
        rbm_input = network.encode(data)
        # Flatten input for RBM
        flat_rbm_input = rbm_input.view(len(rbm_input), (self.levels[network.level]['rbm_visible_units'] ** 2) * self.levels[network.level]['encoder_channels'])

        # Compare data with existing models
        return network.rbm.is_familiar(flat_rbm_input, provide_value=provide_value)

    def train_new_network(self, data, level, target):
        network = self.create_new_model(level, target)
        self.model = network
        self.model.train()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_settings['encoder_learning_rate'])

        for i in range(2):
            # Encode the image
            rbm_input = self.model.encode(data)
            # Flatten input for RBM
            flat_rbm_input = rbm_input.view(len(rbm_input), (self.levels[level]['rbm_visible_units'] ** 2) * self.levels[level]['encoder_channels'])

            # Train RBM
            self.model.rbm.contrastive_divergence(flat_rbm_input, update_weights=True)
            # Encode the image

            # Train encoder
            hidden = self.model.rbm.sample_hidden(flat_rbm_input)
            visible = self.model.rbm.sample_visible(hidden).reshape((
                data.shape[0],
                self.levels[level]['encoder_channels'],
                self.levels[level]['rbm_visible_units'],
                self.levels[level]['rbm_visible_units']
            ))
            loss = self.loss_function(visible, rbm_input)
            loss.backward(retain_graph=True)
            self.model.rbm.calculate_energy_threshold(flat_rbm_input)

        return network

    def _joint_training(self, data, model, depth, target):
        if depth <= 0:
            return

        # Split data into finer regions
        regions = self.divide_data_in_five(data)
        regions_to_train = []
        for region in regions:
            familiar = 0
            for child_model in model.child_networks:
                is_familiar = self.is_familiar(child_model, region)
                if is_familiar:
                    self._joint_training(region, child_model, depth - 1, target)
                    familiar = 1
                    # break
            if familiar == 0:
                regions_to_train.append(region)
        # Train new children if region not recognized
        for region in regions_to_train:
            new_model = self.train_new_network(region, level=model.level + 1, target=target)
            self._joint_training(region, new_model, depth - 1, target)
            model.child_networks.append(new_model)

    def joint_training(self):
        counter = 0
        new_models = self.models
        # new_models = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Assume we have batch size of 1
            data = data.to(self.device)
            target = target.cpu().detach().numpy()[0]

            counter += 1
            if counter % 1 == 0:
                print("______________")
                print("Iteration: ", counter)

                models_counter = np.zeros(self.n_levels, dtype=np.int)
                models_counter[0] = len(self.models)
                for m_1 in self.models:
                    models_counter[1] += len(m_1.child_networks)
                    for m_2 in m_1.child_networks:
                        if self.n_levels == 2:
                            break
                        models_counter[2] += len(m_2.child_networks)
                        for m_3 in m_2.child_networks:
                            if self.n_levels == 3:
                                break
                            models_counter[3] += len(m_3.child_networks)
                            for m_4 in m_3.child_networks:
                                if self.n_levels == 4:
                                    break
                                models_counter[4] += len(m_4.child_networks)

                for i in range(models_counter.shape[0]):
                    print("Level {}: {}".format(i + 1, models_counter[i]))

            n_familiar = 0
            for m in new_models:
                familiar = self.is_familiar(m, data)
                if familiar:
                    n_familiar += 1
                    self._joint_training(data, m, self.n_levels - 1, target)

                if n_familiar >= self.model_settings['min_familiarity_threshold']:
                    break
            if n_familiar >= self.model_settings['min_familiarity_threshold']:
                continue

            model = self.train_new_network(data, level=0, target=target)
            new_models.append(model)
            self._joint_training(data, model, self.n_levels - 1, target)
        # self.models += new_models
