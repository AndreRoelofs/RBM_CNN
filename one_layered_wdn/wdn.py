import torch
from torch import nn
from one_layered_wdn.node import Node
from one_layered_wdn.helpers import *
from torch.nn import functional as F
from torchvision.transforms.functional import resize, center_crop, gaussian_blur


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
        self.models.append(network)
        network.to(self.device)

        return network

    def resize_image(self, image):
        return resize(image, [self.model_settings['image_input_size'], self.model_settings['image_input_size']])

    def loss_function(self, recon_x, x):
        return F.mse_loss(x, recon_x)

    def joint_training(self):
        # torch.autograd.set_detect_anomaly(True)
        counter = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            # Assume we have batch size of 1
            data = data.to(self.device)

            a_n_models = len(self.models)

            counter += 1
            if counter % 25 == 0:
                print("Iteration: ", counter)
                print("n_models", a_n_models)

            n_familiar = 0
            model_counter = 0
            for m in self.models:
                # Encode the image
                rbm_input = m.encode(data)
                # Flatten input for RBM
                flat_rbm_input = rbm_input.view(len(rbm_input), self.model_settings['rbm_visible_units'])

                # Compare data with existing models
                familiar = m.rbm.is_familiar(flat_rbm_input, provide_value=False)
                if familiar >= self.model_settings['min_familiarity_threshold']:
                    n_familiar += 1
                if n_familiar >= self.model_settings['min_familiarity_threshold'] or n_familiar + (
                        a_n_models - model_counter) < self.model_settings['min_familiarity_threshold']:
                    break
                model_counter += 1
            if n_familiar >= self.model_settings['min_familiarity_threshold']:
                # break
                continue

            # If data is unfamiliar, create a new network
            network = self.create_new_model()
            self.model = network
            self.model.train()

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_settings['encoder_learning_rate'])

            for i in range(50):
                # Encode the image
                rbm_input = self.model.encode(data)
                # Flatten input for RBM
                flat_rbm_input = rbm_input.view(len(rbm_input), self.model_settings['rbm_visible_units'])



                familiarity = self.model.rbm.is_familiar(flat_rbm_input, provide_value=False)
                if familiarity == data.shape[0]:
                    self.model.rbm.calculate_energy_threshold(flat_rbm_input)
                    break

                # Train RBM
                self.model.rbm.contrastive_divergence(flat_rbm_input, update_weights=True)

                # Train encoder
                if i % 5 == 0:
                    hidden = self.model.rbm.sample_hidden(flat_rbm_input)
                    visible = self.model.rbm.sample_visible(hidden).reshape((
                        data.shape[0],
                        self.model_settings['encoder_channels'],
                        self.model_settings['image_input_size'],
                        self.model_settings['image_input_size']
                    ))
                    loss = self.loss_function(visible, rbm_input)
                    loss.backward(retain_graph=True)
                    self.model.rbm.calculate_energy_threshold(flat_rbm_input)
