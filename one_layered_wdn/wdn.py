import torch
from torch import nn
from one_layered_wdn.node import Node
from one_layered_wdn.helpers import *
from torch.nn import functional as F
from torchvision.transforms.functional import resize, crop
import matplotlib.pyplot as plt


class WDN(nn.Module):
    def __init__(self, model_settings):
        super().__init__()
        self.device = torch.device("cuda")

        self.models = []
        self.model_settings = model_settings
        # self.create_new_model()

        self.log_interval = 100

    def create_new_model(self):
        network = Node(
            image_channels=self.model_settings['image_channels'],
            encoder_channels=self.model_settings['encoder_channels'],
            rbm_visible_units=self.model_settings['rbm_visible_units'],
            rbm_hidden_units=self.model_settings['rbm_hidden_units'],
            rbm_learning_rate=self.model_settings['rbm_learning_rate'],
            use_relu=self.model_settings['rbm_activation'] == RELU_ACTIVATION
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
            crop(data, 0, 0, 4, 4),
            crop(data, 0, 4, 4, 4),
            crop(data, 4, 0, 4, 4),
            crop(data, 4, 4, 4, 4),
        ]
        return regions

    def is_familiar(self, network, data):
        # Encode the image
        rbm_input = network.encode(data)
        # Flatten input for RBM
        flat_rbm_input = rbm_input.view(len(rbm_input), self.model_settings['rbm_visible_units'])
        # Compare data with existing models
        return network.rbm.is_familiar(flat_rbm_input, provide_value=False)

    def train_new_network(self, data):


    def joint_training(self):
        for batch_idx, (data, _) in enumerate(self.train_loader):
            # Assume we have batch size of 1
            data = data.to(self.device)

            n_familiar = 0
            model_counter = 0
            for m in self.models:
                familiar = self.is_familiar(m, data)

                if familiar == 1:
                    second_level_regions = self.generate_second_level_regions(data)
                    for second_level_region in second_level_regions:
                        for m_second_level in m.child_networks:
                            second_level_familiar = self.is_familiar(m_second_level, second_level_region)

                            if second_level_familiar == 1:
                                third_level_regions = self.generate_third_level_regions(second_level_region)
                                third_level_familiarity = 0
                                for third_level_region in third_level_regions:
                                    third_level_familiar = 0
                                    for m_third_level in m_second_level.child_networks:
                                        third_level_familiar = self.is_familiar(m_third_level, third_level_region)
                                        third_level_familiarity += third_level_familiar

                                        if third_level_familiar == 1:
                                            break
                                    if not third_level_familiar:











