import torch
from torch import nn
from one_layered_wdn.node import Node
from one_layered_wdn.helpers import *
from torch.nn import functional as F
from torchvision.transforms.functional import resize, crop, rotate, affine, center_crop
import matplotlib.pyplot as plt


class WDN(nn.Module):
    def __init__(self, model_settings):
        super().__init__()
        self.device = torch.device("cuda")

        self.models = []
        self.model_settings = model_settings
        # self.create_new_model()

        self.levels = [
            {'rbm_visible_units': 14, 'rbm_learning_rate': 1e-20},
            {'rbm_visible_units': 7, 'rbm_learning_rate': 1e-20},
            {'rbm_visible_units': 3, 'rbm_learning_rate': 1e-20},
        ]

        self.n_levels = 3

        self.log_interval = 100

    def create_new_model(self, level):

        hidden_units = self.model_settings['rbm_hidden_units']
        if level == 2:
            hidden_units = 1
        network = Node(
            image_channels=1,
            encoder_channels=1,
            rbm_visible_units=self.levels[level]['rbm_visible_units'] ** 2,
            rbm_hidden_units=hidden_units,
            rbm_learning_rate=self.levels[level]['rbm_learning_rate'],
            use_relu=self.model_settings['rbm_activation'] == RELU_ACTIVATION,
            level=level
        )
        network.to(self.device)

        return network

    def resize_image(self, image):
        return resize(image, [self.model_settings['image_input_size'], self.model_settings['image_input_size']])

    def loss_function(self, recon_x, x):
        return F.mse_loss(x, recon_x)

    def generate_second_level_regions(self, data):
        regions = [
            crop(data, 0, 0, 7, 7),
            crop(data, 0, 7, 7, 7),
            crop(data, 7, 0, 7, 7),
            crop(data, 7, 7, 7, 7),
        ]
        return regions

    def generate_third_level_regions(self, data):
        regions = [
            crop(data, 0, 0, 3, 3),
            crop(data, 0, 4, 3, 3),
            crop(data, 4, 0, 3, 3),
            crop(data, 4, 4, 3, 3),
        ]
        return regions

    def divide_data_in_four(self, data):
        original_size = data.shape[-1]
        # Accepts images of up to 256x256 size, more needed?
        new_size = torch.floor_(original_size / 2).type_as(torch.uint8).numpy()
        new_size = max(new_size, 2)

        regions = [
            crop(data, 0, 0, new_size, new_size),
            crop(data, 0, new_size, new_size, new_size),
            crop(data, new_size, 0, new_size, new_size),
            crop(data, new_size, new_size, new_size, new_size),
        ]

        return regions

    def is_familiar(self, network, data, provide_value=False, provideInput=False):
        # Encode the image
        rbm_input = network.encode(data)
        # Flatten input for RBM
        flat_rbm_input = rbm_input.view(len(rbm_input), self.levels[network.level]['rbm_visible_units'] ** 2)

        if provideInput:
            return network.rbm.is_familiar(flat_rbm_input, provide_value=provide_value), rbm_input

        # Compare data with existing models
        return network.rbm.is_familiar(flat_rbm_input, provide_value=provide_value)

    def train_new_network(self, data, level):
        network = self.create_new_model(level)
        self.model = network
        self.model.train()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_settings['encoder_learning_rate'])

        for i in range(2):
            # Encode the image
            rbm_input = self.model.encode(data)
            # Flatten input for RBM
            flat_rbm_input = rbm_input.view(len(rbm_input), self.levels[level]['rbm_visible_units'] ** 2)

            # Train RBM
            self.model.rbm.contrastive_divergence(flat_rbm_input, update_weights=True)
            # Encode the image

            # Train encoder
            hidden = self.model.rbm.sample_hidden(flat_rbm_input)
            visible = self.model.rbm.sample_visible(hidden).reshape((
                data.shape[0],
                self.model_settings['encoder_channels'],
                self.levels[level]['rbm_visible_units'],
                self.levels[level]['rbm_visible_units']
            ))
            loss = self.loss_function(visible, rbm_input)
            loss.backward(retain_graph=True)
            self.model.rbm.calculate_energy_threshold(flat_rbm_input)

        return network

    def _joint_training(self, data, model, depth):
        if depth == 0:
            return

        is_region_familiar = False
        # Split data into finer regions
        regions = self.divide_data_in_four(data)
        for region in regions:
            familiar = 0
            for child_model in model.child_networks:
                if self.is_familiar(child_model, region):
                    familiar += 1
                    is_region_familiar = True
                    self._joint_training(region, child_model, depth - 1)
                    # TODO: remove and see what happens
                    break
            # Train new children if not recognized
            if familiar == 0:
                child_model = self.train_new_network(region, level=model.level+1)
                self._joint_training(region, child_model, depth - 1)
                model.child_networks.append(child_model)
                is_region_familiar = True

        return is_region_familiar

    def joint_training(self):
        counter = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            # Assume we have batch size of 1
            data = data.to(self.device)

            a_n_models = len(self.models)

            counter += 1
            if counter % 50 == 0:
                print("______________")
                print("Iteration: ", counter)
                print("n_models", a_n_models)
                n_second_level_children = 0
                n_third_level_children = 0

                for m in self.models:
                    n_second_level_children += len(m.child_networks)
                    for m_child in m.child_networks:
                        n_third_level_children += len(m_child.child_networks)

                print("n_second_models", n_second_level_children)
                print("n_third_models", n_third_level_children)

            n_familiar = 0
            for m in self.models:
                familiar, first_level_input = self.is_familiar(m, data, provideInput=True)
                # if True:
                if familiar == 1:
                    n_familiar += 1
                    second_level_regions = self.generate_second_level_regions(first_level_input)
                    for second_level_region in second_level_regions:
                        second_level_familiar = 0
                        for m_second_level in m.child_networks:
                            second_level_familiar, second_level_input = self.is_familiar(m_second_level,
                                                                                         second_level_region,
                                                                                         provideInput=True)
                            if second_level_familiar == 1:
                                third_level_regions = self.generate_third_level_regions(second_level_input)

                                for third_level_region in third_level_regions:
                                    third_level_familiar = 0
                                    for m_third_level in m_second_level.child_networks:
                                        third_level_familiar += self.is_familiar(m_third_level, third_level_region)

                                        if third_level_familiar == 1:
                                            break
                                    if third_level_familiar == 0:
                                        third_level_child = self.train_new_network(third_level_region, level=2)
                                        m_second_level.child_networks.append(third_level_child)
                                break
                        if second_level_familiar == 0:
                            second_level_child = self.train_new_network(second_level_region, level=1)
                            m.child_networks.append(second_level_child)
                            third_level_regions = self.generate_third_level_regions(second_level_region)
                            for third_level_region in third_level_regions:
                                third_level_child = self.train_new_network(third_level_region, level=2)
                                second_level_child.child_networks.append(third_level_child)
                if n_familiar >= self.model_settings['min_familiarity_threshold']:
                    break
            if n_familiar >= self.model_settings['min_familiarity_threshold']:
                return counter
            network = self.train_new_network(data, level=0)
            self.models.append(network)
            second_level_regions = self.generate_second_level_regions(data)
            for second_level_region in second_level_regions:
                second_level_child = self.train_new_network(second_level_region, level=1)
                network.child_networks.append(second_level_child)
                third_level_regions = self.generate_third_level_regions(second_level_region)
                for third_level_region in third_level_regions:
                    third_level_child = self.train_new_network(third_level_region, level=2)
                    second_level_child.child_networks.append(third_level_child)
            return counter
