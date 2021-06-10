import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import resize, crop, rotate, affine, center_crop, five_crop, ten_crop, vflip, \
    hflip
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from one_layered_wdn.node import Node
from one_layered_wdn.helpers import *
import random
import wandb


class WDN(nn.Module):
    def __init__(self, model_settings):
        super().__init__()
        self.device = torch.device("cuda")

        self.models = []
        self.model_settings = model_settings

        self.levels = model_settings['levels_info']

        self.n_levels = self.model_settings['n_levels']
        self.use_relu = self.model_settings['use_relu']
        self.levels_counter = np.zeros(self.n_levels)
        self.debug = False
        self.train_data = None
        self.test_data = None
        self.models_total = 0

        self.used_ids = []

    def create_new_model(self, level, target):

        settings = self.levels[level]

        network = Node(
            image_channels=settings['input_channels'],
            encoder_channels=settings['encoder_channels'],
            encoder_weight_mean=settings['encoder_weight_mean'],
            encoder_weight_variance=settings['encoder_weight_variance'],
            rbm_weight_mean=settings['rbm_weight_mean'],
            rbm_weight_variance=settings['rbm_weight_variance'],
            rbm_visible_units=(settings['rbm_visible_units']) * settings['encoder_channels'],
            rbm_hidden_units=settings['rbm_hidden_units'],
            use_relu=self.use_relu,
            level=level
        )
        network.target = target
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

        # return five_crop(data, [new_size, new_size])
        return ten_crop(data, [new_size, new_size])

    def _calculate_number_of_children(self, network):
        if len(network.child_networks) == 0:
            network.n_children = 1
            return
        for child_network in network.child_networks:
            self._calculate_number_of_children(child_network)
            network.n_children += child_network.n_children

    def _reset_number_of_children(self, network):
        for child in network.child_networks:
            self._reset_number_of_children(child)
        network.n_children = 0

    def calculate_number_of_children(self):
        self.models_total = 0
        for m in self.models:
            self._reset_number_of_children(m)
            self._calculate_number_of_children(m)
            self.models_total += m.n_children

    def is_familiar(self, network, data, provide_value=False, provide_encoding=False):
        # Encode the image
        rbm_input = network.encode(data)
        # Flatten input for RBM
        flat_rbm_input = rbm_input.view(len(rbm_input), (self.levels[network.level]['rbm_visible_units']) *
                                        self.levels[network.level]['encoder_channels'])

        if provide_encoding:
            return network.rbm.is_familiar(flat_rbm_input, provide_value=provide_value), rbm_input

        recon_error = np.sum((network.rbm.energy_threshold -
                              F.mse_loss(
                                  flat_rbm_input,
                                  network.rbm(flat_rbm_input),
                                  reduction='none')).cpu().detach().numpy(), axis=1)

        # recon_error = np.sum((network.rbm.energy_threshold -
        #                       F.mse_loss(
        #                           network.rbm(flat_rbm_input),
        #                           flat_rbm_input,
        #                           reduction='none')).cpu().detach().numpy(), axis=1)

        if provide_value:
            return recon_error, recon_error >= 0.0,
        return recon_error >= 0.0
        # Compare data with existing models
        # return network.rbm.is_familiar(flat_rbm_input, provide_value=provide_value)

    def train_new_network(self, data, level, target, provide_encoding=False, first_training=True, network=None,
                          train=True, update_threshold=True):
        if network is None:
            network = self.create_new_model(level, target)
            network.train()
            self.levels_counter[level] += 1
            self.models_total += 1

        if train:
            lr = self.levels[network.level]['encoder_learning_rate']
            if not update_threshold:
                lr = 1e-5
            encoder_optimizer = torch.optim.Adam(network.encoder.parameters(),
                                                 lr=lr)
            # encoder_optimizer = torch.optim.SGD(network.encoder.parameters(),
            #                                      lr=lr)
            n_train_epochs = self.levels[level]['n_training']
            if not first_training:
                n_train_epochs = self.levels[level]['n_training_second']
            for i in range(n_train_epochs):
                # Encode the image
                rbm_input = network.encode(data)
                # Flatten input for RBM
                flat_rbm_input = rbm_input.detach().clone().view(len(rbm_input),
                                                                 (self.levels[level]['rbm_visible_units']) *
                                                                 self.levels[level]['encoder_channels'])
                # if i % 5 == 0:
                if i % 10 == 0 and i != 0:
                # if i == 0:
                    network.rbm.contrastive_divergence(flat_rbm_input)

                # Train encoder
                rbm_output = network.rbm(flat_rbm_input)
                encoder_loss = network.encoder.loss_function(rbm_output.detach().clone().reshape(rbm_input.shape),
                                                             rbm_input)
                # encoder_loss = network.encoder.loss_function(rbm_input,
                #                                              rbm_output.detach().clone().reshape(rbm_input.shape))
                encoder_optimizer.zero_grad()
                encoder_loss.backward(retain_graph=True)
                encoder_optimizer.step()

        # if False:
        if first_training:
            rbm_input = network.encode(data)
            flat_rbm_input = rbm_input.clone().detach().view(len(rbm_input),
                                                             (self.levels[0]['rbm_visible_units']) *
                                                             self.levels[0]['encoder_channels'])
            rbm_output = network.rbm(flat_rbm_input)
            network.rbm.energy_threshold = F.mse_loss(flat_rbm_input, rbm_output)
            # network.rbm.energy_threshold = F.mse_loss(rbm_output, flat_rbm_input)
            for _ in range(1):
                image_energies = self.calculate_energies(network)
                # Only get activated images
                image_energies = image_energies[image_energies[:, 3] == 1]
                # Only get images of target class
                # image_energies = image_energies[image_energies[:, 1] == target]

                image_idx = image_energies[:, 2].astype(np.int)

                # image_idx = image_idx[image_idx not in np.array(self.used_ids)]

                if len(image_idx) == 0:
                    break
                # else:
                #     image_idx = image_idx[0]
                # if len(image_idx) == 0:
                #     break


                train_loader = torch.utils.data.DataLoader(
                    self.train_data,
                    batch_size=min(len(image_idx), 100),
                    shuffle=False,
                    sampler=SubsetRandomSampler(image_idx)
                )

                for batch_idx, (data, target) in enumerate(train_loader):
                    data = data.to(self.device)
                    self.train_new_network(data, level, target, network=network, first_training=False, train=True,
                                           update_threshold=False)
                    self.train_new_network(data, level, target, network=network, first_training=False, train=False,
                                           update_threshold=True)

        if update_threshold:
            rbm_input = network.encode(data)
            flat_rbm_input = rbm_input.clone().detach().view(len(rbm_input),
                                                             (self.levels[0]['rbm_visible_units']) *
                                                             self.levels[0]['encoder_channels'])
            rbm_output = network.rbm(flat_rbm_input)
            network.rbm.energy_threshold = F.mse_loss(flat_rbm_input, rbm_output)
            # network.rbm.energy_threshold = F.mse_loss(rbm_output, flat_rbm_input)


        if provide_encoding:
            return network, network.encode(data)
        return network

    def calculate_energies(self, node, use_training_data=True):
        if use_training_data:
            train_loader = torch.utils.data.DataLoader(
                # np.setdiff1d(self.train_data, self.train_data[self.used_ids]),
                self.train_data,
                batch_size=5000,
                shuffle=False,
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                self.test_data,
                batch_size=5000,
                shuffle=False,
            )

        image_energies = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(self.device)
            distances, activations = self.is_familiar(node, data, provide_value=True)

            for i in range(len(distances)):
                d = distances[i]
                t = target[i]
                image_energies.append([d, t.numpy().astype(int), 5000 * batch_idx + i, activations[i]])
        image_energies = np.array(image_energies)
        return image_energies[image_energies[:, 0].argsort()]

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
                    break
            if familiar == 0:
                regions_to_train.append(region)
        # Train new children if region not recognized
        new_models = []
        for region in regions_to_train:
            is_familiar = 0
            for m in new_models:
                is_familiar = self.is_familiar(m, region)
                if is_familiar == 1:
                    break
            if is_familiar == 1:
                continue
            n_tries = 5
            activation_threshold = 60
            n_activations = 0
            while n_tries != 0:
                n_tries -= 1
                new_model = self.train_new_network(region, level=model.level + 1, target=target)
                image_energies = self.calculate_energies(new_model)
                target_digit_indices = [10000 - (i + 1) for i, e in reversed(list(enumerate(image_energies))) if
                                        int(e[1]) == target]
                n_activations = sum([i < 100 for i in target_digit_indices])
                if n_activations >= activation_threshold:
                    break
            if n_activations >= activation_threshold:
                new_models.append(new_model)
                self._joint_training(region, new_model, depth - 1, target)
                model.child_networks.append(new_model)

    def joint_training(self):
        counter = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Assume we have batch size of 1
            data = data.to(self.device)
            target = target.cpu().detach().numpy()[0]

            counter += 1
            if counter % self.model_settings['log_interval'] == 0:
                print("Iteration: ", counter)

                models_counter = np.zeros(self.n_levels, dtype=np.int)
                models_counter[0] = len(self.models)
                for m_1 in self.models:
                    if self.n_levels == 1:
                        break
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
                print("______________")

            n_familiar = 0
            for m in self.models:
                familiar = self.is_familiar(m, data)
                if familiar:
                    n_familiar += 1
                    # for current_data in [data, hflip(data)]:
                    # self._joint_training(data, m, self.n_levels - 1, target)

                if n_familiar >= self.model_settings['min_familiarity_threshold']:
                    break
            if n_familiar >= self.model_settings['min_familiarity_threshold']:
                continue

            n_tries = 5
            activation_threshold = 100
            score = 0
            model = None
            models = []
            while n_tries != 0:
                n_tries -= 1
                model = self.train_new_network(data, level=0, target=target)
                image_energies = self.calculate_energies(model, use_training_data=False)

                # target_digit_indices = [len(image_energies) - (i + 1) for i, e in reversed(list(enumerate(image_energies))) if
                #                         int(e[1]) == target]
                #
                # one_hundred_test = sum([i < 100 for i in target_digit_indices])

                # score = max(one_hundred_test,  score)
                # score = 1
                #
                t_energies = image_energies[image_energies[:, 3] == 1]

                # t_energies = t_energies[t_energies[:, 2] not in np.array(self.used_ids)]

                # if t_energies.shape[0] > 0:
                #     t_energies = t_energies[0]
                #
                # t_energies = t_energies[t_energies[:, 1] == target]
                #
                # n_activations = t_energies.shape[0]
                score = t_energies.shape[0]
                # score = 0

                # target_digit_indices = [10000 - (i + 1) for i, e in reversed(list(enumerate(image_energies))) if
                #                         int(e[1]) == target]
                # n_activations = sum([i < 100 for i in target_digit_indices])
                print('Score value: {}'.format(score))
                # print('N activations: {}'.format(n_activations))
                # if len(t_energies) == 0:
                #     models.append([model, score, np.array([])])
                # else:
                #     models.append([model, score, t_energies[:, 2]])
                #
                if score >= activation_threshold:
                    models.append([model, score, t_energies[:, 2]])
                    break
            # if score >= activation_threshold:
            if len(models) != 0:
                models = np.array(models)
                models = models[(-models[:, 1]).argsort()]
                model = models[0][0]
                self.models.insert(0, model)

                # self.used_ids += models[0][2].astype(int).tolist()
                # print("Length of used_ids: {}".format(len(set(self.used_ids))))

            # else:
            #     models = np.array(models)
            #     models = models[(-models[:, 1]).argsort()]
            #     model = models[0][0]
            #     self.models.insert(0, model)
            #     self._joint_training(data, model, self.n_levels - 1, target)


def train_wdn(train_data, test_data, settings, wbc=None, model=None):
    if model is None:
        model = WDN(settings)
    model.train_data = train_data
    model.test_data = test_data
    for i in range(10):
        # for i in [5]:
        print("Training digit: ", i)
        subset_indices = (torch.tensor(train_data.targets) == i).nonzero().view(-1)

        model.train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=1,
            shuffle=False,
            sampler=SubsetRandomSampler(subset_indices)
        )
        model.joint_training()
        log_dict = {'n_models': model.models_total}
        n_parameters = 0
        for model_level in range(model.n_levels):
            log_dict['n_models_lvl_{}'.format(model_level + 1)] = model.levels_counter[model_level]
            # (kernel_size * encoder_channels) + rbm parameters
            n_parameters += ((9 * model.levels[model_level]['encoder_channels']) + model.levels[model_level][
                'rbm_visible_units'] + model.levels[model_level]['rbm_hidden_units']) * model.levels_counter[
                                model_level]
        log_dict['n_parameters'] = n_parameters
        if wbc is not None:
            wandb.log(log_dict)
        random.shuffle(model.models)

    model.calculate_number_of_children()

    models_counter = np.zeros(model.n_levels, dtype=np.int)
    models_counter[0] = len(model.models)
    for m_1 in model.models:
        if model.n_levels == 1:
            break
        models_counter[1] += len(m_1.child_networks)
        for m_2 in m_1.child_networks:
            if model.n_levels == 2:
                break
            models_counter[2] += len(m_2.child_networks)
            for m_3 in m_2.child_networks:
                if model.n_levels == 3:
                    break
                models_counter[3] += len(m_3.child_networks)
                for m_4 in m_3.child_networks:
                    if model.n_levels == 4:
                        break
                    models_counter[4] += len(m_4.child_networks)

    for i in range(models_counter.shape[0]):
        print("Level {}: {}".format(i + 1, models_counter[i]))

    log_dict = {'n_models': model.models_total}
    n_parameters = 0
    for model_level in range(model.n_levels):
        log_dict['n_models_lvl_{}'.format(model_level + 1)] = model.levels_counter[model_level]
        # (kernel_size * encoder_channels) + rbm parameters
        n_parameters += ((9 * model.levels[model_level]['encoder_channels']) + model.levels[model_level][
            'rbm_visible_units'] + model.levels[model_level]['rbm_hidden_units']) * model.levels_counter[model_level]
    log_dict['n_parameters'] = n_parameters

    if wbc is not None:
        wandb.log(log_dict)

    return model
