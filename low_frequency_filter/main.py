# Import packages
import numpy as np
import torch
from scipy.stats import norm
from torchvision.datasets import FashionMNIST, MNIST, CIFAR10
import wandb
from one_layered_wdn.helpers import *
from one_layered_wdn.wdn import WDN
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler
import matplotlib.pyplot as plt
import random
import time
from sys import exit

# %% Initialize hyperparameters
# General
data_path = "../data"
dataset_name = FASHIONMNIST_DATASET
# dataset_name = MNIST_DATASET
# dataset_name = CIFAR10_DATASET

ten_test_history = []
one_hundred_test_history = []
five_hundred_test_history = []

# RBM settings
# n_rbm_training = 1
# n_filter_training = 2

# Encoder settings
n_sequential_filters = 1
baseline_run = True

# Dataset settings
train_data = None
test_data = None
image_size = None
input_filters = None
n_train_data = None
n_test_data = None
n_classes = None
# target_digit = None
target_digit = 0
target_digits = []
target_classes = None

# custom_seed = np.random.randint(0, 100000)
# print("Seed: {}".format(custom_seed))


# reset_seed(seed=0)

# %% Load data
if dataset_name == FASHIONMNIST_DATASET:
    image_size = 28
    input_filters = 1
    n_train_data = 60000
    n_test_data = 10000
    n_classes = 10
    target_classes = fashion_mnist_classes

    train_data = FashionMNIST(data_path, train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  # transforms.Normalize((0.1307,), (0.3081,)),
                              ]))

    test_data = FashionMNIST(data_path, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,)),
    ]))

if dataset_name == MNIST_DATASET:
    image_size = 28
    input_filters = 1
    n_train_data = 60000
    n_test_data = 10000
    n_classes = 10
    target_classes = np.arange(0, 10)

    train_data = MNIST(data_path, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           # transforms.Normalize((0.1307,), (0.3081,)),
                       ]))

    test_data = MNIST(data_path, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,)),
    ]))

if dataset_name == CIFAR10_DATASET:
    image_size = 32
    input_filters = 1
    n_train_data = 50000
    n_test_data = 10000
    n_classes = 10
    target_classes = cifar10_classes

    train_data = CIFAR10(data_path, train=True, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Grayscale(),
                             # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                         ]))

    test_data = CIFAR10(data_path, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))

# n_train_data = 10
# n_test_data = 10

train_data.data = train_data.data[:n_train_data]
train_data.targets = train_data.targets[:n_train_data]

test_data.data = test_data.data[:n_test_data]
test_data.targets = test_data.targets[:n_test_data]
# %% Wandb
# Disable wandb
wbc = None

# Enable wandb
wandb.init(project='LF_mean_std_{}'.format(dataset_name))
wbc = wandb.config
# wandb.run.name = '{}_{}'.format(target_classes[target_digit], wandb.run.name)
# wandb.run.name = '{}_{}'.format(target_classes[target_digit], 'one_shot')
wandb.run.name = '{}_{}'.format(target_classes[target_digit], 'unsupervised_multi_shot')
# wandb.run.name = '{}_{}'.format(target_classes[target_digit], 'supervised_multi_shot')

# %% Configure and init WDN
wdn_settings = {
    'model_name': 'test',
    'n_levels': 1,

    'image_input_size': image_size,
    'image_channels': input_filters,

    'min_familiarity_threshold': 1,
    'log_interval': 50,
    'use_relu': False,

    'levels_info': [
        {
            'input_channels': input_filters, 'encoder_channels': 1, 'rbm_visible_units': image_size ** 2,
            'encoder_weight_mean': 0.1, 'encoder_weight_variance': 0.01,
            'rbm_weight_mean': 0.0, 'rbm_weight_variance': 0.01,
            'rbm_hidden_units': 300, 'encoder_learning_rate': 1e-3,
            'n_training': 50, 'n_training_second': 2,
        },
    ]
}

wdn = WDN(wdn_settings)


# %% Define helpers
def plot_energies(energies, record_tests=True, use_train_data=True):
    if use_train_data:
        n_data = n_train_data
    else:
        n_data = n_test_data

    target_digit_indices = [n_data - (i + 1) for i, e in reversed(list(enumerate(energies))) if
                            int(e[1]) == target_digit]

    ten_test = sum([i < 10 for i in target_digit_indices])
    one_hundred_test = sum([i < 100 for i in target_digit_indices])
    five_hundred_test = sum([i < 500 for i in target_digit_indices])

    print("10 test: {}".format(ten_test))
    print("100 test: {}".format(one_hundred_test))
    print("500 test: {}".format(five_hundred_test))

    if record_tests:
        ten_test_history.append([target_digit, ten_test])
        one_hundred_test_history.append([target_digit, one_hundred_test])
        five_hundred_test_history.append([target_digit, five_hundred_test])

    # return

    min_energy = energies[:, 0].min()
    max_energy = energies[:, 0].max()
    n_bins = 20
    rows = 5
    columns = 2
    #
    f, axarr = plt.subplots(rows, columns)

    for y in range(rows):
        for x in range(columns):
            class_id = y * columns + x
            class_energies = energies[energies[:, 1] == class_id][:, 0]
            mu = class_energies.mean()
            std = class_energies.std()
            title = '{} mu: {} std: {}'.format(
                target_classes[class_id],
                round(mu, 2),
                round(std, 2),
            )
            if class_id == target_digit:
                title = 'T: {}'.format(title)

            axarr[y, x].set_title(title)

            axarr[y, x].hist(class_energies, bins=n_bins, density=True, alpha=0.6)
            axarr[y, x].set_xlim([min_energy, max_energy])
            # axarr[y, x].set_xlim([-200, 0])

            x_axis = np.linspace(min_energy, max_energy, 100)
            p = norm.pdf(x_axis, mu, std)
            axarr[y, x].plot(x_axis, p, 'k', linewidth=2, alpha=0.8, color='g')

    plt.show()

    # return

    # plt.imshow(train_data.data[random_tr_indice].numpy().reshape((image_size, image_size)), cmap='gray')
    # plt.show()
    # plt.imshow(train_data.data[similar_image_idx].numpy().reshape((image_size, image_size)), cmap='gray')
    # plt.show()

    similar_image_idx = int(energies[::-1][0][2])
    # similar_image_idx = int(energies[1][2])

    generator = torch.Generator()
    generator.manual_seed(0)

    if random_tr_indice == similar_image_idx:
        similar_image_idx = int(energies[::-1][1][2])

    wdn.train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=2,
        shuffle=False,
        sampler=SubsetRandomSampler([random_tr_indice, similar_image_idx], generator)
    )
    for batch_idx, (data, target) in enumerate(wdn.train_loader):
        data = data.to(wdn.device)
        rbm_input = node.encode(data)

        flat_rbm_input = rbm_input.clone().detach().view(len(rbm_input),
                                                         (wdn.levels[0]['rbm_visible_units']) *
                                                         wdn.levels[0]['encoder_channels'])

        rbm_output = node.rbm(flat_rbm_input)

        og_encoding, similar_encoding = rbm_input
        og_recon, similar_recon = rbm_output

        f, axarr = plt.subplots(3, 2)

        fig_title = 'One Shot {}'.format(
            target_classes[target_digit],
        )
        fig_subtitle = '10 Test: {}; 100 Test: {}; 500 Test: {};'.format(
            ten_test,
            one_hundred_test,
            five_hundred_test,
        )

        f.suptitle(fig_title, fontsize=16)
        f.text(x=0.33, y=0.05, s=fig_subtitle, fontsize=8)

        axarr[0, 0].set_title('Original image')
        axarr[0, 1].set_title('Similar image')
        axarr[1, 0].set_title('Original image encoded')
        axarr[1, 1].set_title('Similar image encoded')
        axarr[2, 0].set_title('Original image recon')
        axarr[2, 1].set_title('Similar image recon')

        axarr[0, 0].imshow(data[0][0].reshape((image_size, image_size)).cpu().detach().numpy(), cmap='gray')
        axarr[0, 1].imshow(data[1][0].reshape((image_size, image_size)).cpu().detach().numpy(), cmap='gray')

        axarr[1, 0].imshow(og_encoding[0].reshape((image_size, image_size)).cpu().detach().numpy(), cmap='gray')
        axarr[1, 1].imshow(similar_encoding[0].reshape((image_size, image_size)).cpu().detach().numpy(), cmap='gray')

        axarr[2, 0].imshow(og_recon.reshape((image_size, image_size)).cpu().detach().numpy(), cmap='gray')
        axarr[2, 1].imshow(similar_recon.reshape((image_size, image_size)).cpu().detach().numpy(), cmap='gray')

        axarr[0, 0].axis('off')
        axarr[0, 1].axis('off')
        axarr[1, 0].axis('off')
        axarr[1, 1].axis('off')
        axarr[2, 0].axis('off')
        axarr[2, 1].axis('off')

        if wbc is None:
            # plt.savefig('images/{}_run_{}.png'.format(fig_title, run_id))
            plt.show()
            # exit(1)
        else:
            wandb.log({'run_plot': f})
            plt.close(f)


def calculate_energies(node, use_train_data=True):
    if use_train_data:
        wdn.train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=min(5000, n_train_data),
            shuffle=False,
        )
    else:
        wdn.train_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=min(5000, n_test_data),
            shuffle=False,
        )

    image_energies = []
    n_activations = 0
    for batch_idx, (data, target) in enumerate(wdn.train_loader):
        data = data.to(wdn.device)
        distances, activations = wdn.is_familiar(node, data, provide_value=True)

        n_activations += activations.sum()

        for i in range(len(distances)):
            d = distances[i]
            t = target[i]
            # 0 - distance, 1 - target, 2 - image index, 3 - recognized by RBM
            image_energies.append([d, t.numpy().astype(int), 5000 * batch_idx + i, activations[i]])
    print("Number of activations: {}".format(n_activations))
    image_energies = np.array(image_energies)
    return image_energies[image_energies[:, 0].argsort()]


def train_node(node, tr_indices, update_threshold=False, train=True):
    wdn.train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=min(len(tr_indices), 100),
        shuffle=False,
        sampler=SubsetRandomSampler(tr_indices)
        # sampler=SubsetRandomSampler(np.random.randint(0, n_train_data, 100))
    )
    for batch_idx, (data, target) in enumerate(wdn.train_loader):
        data = data.to(wdn.device)

        if train:
            encoder_optimizer = torch.optim.Adam(node.encoder.parameters(),
                                                 lr=wdn.levels[node.level]['encoder_learning_rate'])
            # encoder_optimizer = torch.optim.SGD(node.encoder.parameters(),
            #                                      lr=wdn.levels[node.level]['encoder_learning_rate'])
            n_iterations = wdn.levels[0]['n_training']
            if update_threshold is False:
                n_iterations = wdn.levels[0]['n_training_second']
            for i in range(n_iterations):
                # Encode the image
                rbm_input = node.encode(data)
                # Flatten input for RBM
                flat_rbm_input = rbm_input.clone().detach().view(len(rbm_input),
                                                                 (wdn.levels[0]['rbm_visible_units']) *
                                                                 wdn.levels[0]['encoder_channels'])
                if i % 5 == 0:
                    # if i == 0:
                    node.rbm.contrastive_divergence(flat_rbm_input)

                # Train encoder
                rbm_output = node.rbm(flat_rbm_input)
                encoder_loss = node.encoder.loss_function(rbm_input,
                                                          rbm_output.clone().detach().reshape(rbm_input.shape))
                encoder_loss.backward(retain_graph=True)
                encoder_optimizer.step()

        if update_threshold:
            rbm_input = node.encode(data)
            flat_rbm_input = rbm_input.clone().detach().view(len(rbm_input),
                                                             (wdn.levels[0]['rbm_visible_units']) *
                                                             wdn.levels[0]['encoder_channels'])
            rbm_output = node.rbm(flat_rbm_input)
            node.rbm.energy_threshold = torch.nn.functional.mse_loss(flat_rbm_input, rbm_output)

            # f, axarr = plt.subplots(1, 3)
            #
            # axarr[0].set_title('Original image')
            # axarr[1].set_title('Trained convolution')
            # axarr[2].set_title('RBM reconstruction')
            #
            # axarr[0].imshow(data[0][0].reshape((image_size, image_size)).cpu().detach().numpy(), cmap='gray')
            # axarr[1].imshow(rbm_input[0].reshape((image_size, image_size)).cpu().detach().numpy(), cmap='gray')
            # axarr[2].imshow(rbm_output[0].reshape((image_size, image_size)).cpu().detach().numpy(), cmap='gray')
            #
            # axarr[0].axis('off')
            # axarr[1].axis('off')
            # axarr[2].axis('off')
            # #
            # plt.show()
    return node


def create_node():
    global target_digit
    global target_digits
    global node

    target = train_data.targets[random_tr_indice]
    # target_digit = target
    target_digits = [target_digit]

    node = wdn.create_new_model(0, target)
    wdn.levels_counter[0] += 1
    wdn.models_total += 1

    node.train()
    node = train_node(node, [random_tr_indice], train=True, update_threshold=True)
    # image_energies = calculate_energies(node)
    # plot_energies(image_energies)

    for _ in range(0):
        image_energies = calculate_energies(node)
        # plot_energies(image_energies)
        # Only get activated images
        image_energies = image_energies[image_energies[:, 3] == 1]
        # Only get images of the same class
        # image_energies = image_energies[image_energies[:, 1] == target_digit]

        if len(image_energies) == 0:
            break

        print('Training on {}'.format(len(image_energies)))

        node = train_node(node, image_energies[:, 2].astype(np.int), train=True)
        node = train_node(node, [random_tr_indice], train=False, update_threshold=True)

    node.eval()
    image_energies = calculate_energies(node, use_train_data=False)
    plot_energies(image_energies, use_train_data=False)

    # image_energies = calculate_energies(node, use_train_data=True)
    # plot_energies(image_energies, use_train_data=True)

    return node


# %% Create random node

# reset_seed()
node = None
for run_id in range(10):
    print("Run {}".format(run_id))
    if target_digit is None:
        random_tr_indice = np.random.randint(0, n_train_data)
    else:
        if isinstance(train_data.targets, list):
            possible_target_indices = np.where(np.array(train_data.targets) == target_digit)[0]
        else:
            possible_target_indices = np.where(train_data.targets == target_digit)[0]
        random_tr_indice = np.random.choice(possible_target_indices)
    node = create_node()

ten_test_history = np.array(ten_test_history)
one_hundred_test_history = np.array(one_hundred_test_history)
five_hundred_test_history = np.array(five_hundred_test_history)

c_hist_10 = ten_test_history[ten_test_history[:, 0] == target_digit][:, 1]
c_hist_100 = one_hundred_test_history[one_hundred_test_history[:, 0] == target_digit][:, 1]
c_hist_500 = five_hundred_test_history[five_hundred_test_history[:, 0] == target_digit][:, 1]

print('Class {}'.format(target_classes[target_digit]))
print('10 test: mean {} std {}'.format(round(c_hist_10.mean(), 2), round(c_hist_10.std(), 2)))
print('100 test: mean {} std {}'.format(round(c_hist_100.mean(), 2), round(c_hist_100.std(), 2)))
print('500 test: mean {} std {}'.format(round(c_hist_500.mean(), 2), round(c_hist_500.std(), 2)))

if wbc is not None:
    wandb.log({
        '10_test_mean_{}'.format(target_classes[target_digit]): c_hist_10.mean(),
        '10_test_std_{}'.format(target_classes[target_digit]): c_hist_10.std(),

        '100_test_mean_{}'.format(target_classes[target_digit]): c_hist_100.mean(),
        '100_test_std_{}'.format(target_classes[target_digit]): c_hist_100.std(),

        '500_test_mean_{}'.format(target_classes[target_digit]): c_hist_500.mean(),
        '500_test_std_{}'.format(target_classes[target_digit]): c_hist_500.std(),
    })

    wandb.finish()
# exit(1)

# %% Conduct 100 tests

# image_energies = calculate_energies(node)
#
# plot_energies(image_energies)
#
# print("target {} {}".format(target_digit, target_classes[target_digit]))
#
# test = 0
