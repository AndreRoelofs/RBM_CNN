import gzip
import numpy as np
import matplotlib.pyplot as plt
import struct
import time
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
import torch
from torch import nn
from autoencoder.c10_model import Autoencoder
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torchvision import transforms

data_path = '../data'

# Load data
train_data = CIFAR10(data_path, train=True, download=True,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         # transforms.Normalize((0.1307,), (0.3081,)),
                     ]))

test_data = CIFAR10(data_path, train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,)),
]))

# train_data.data = train_data.data[:100]
# train_data.targets = train_data.targets[:100]
#
# test_data.data = test_data.data[:100]
# test_data.targets = test_data.targets[:100]

# train_data = CIFAR10(data_path, train=True, download=True,
#                      transform=transforms.Compose([
#                          transforms.ToTensor(),
#                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#                      ]))
#
# test_data = CIFAR10(data_path, train=False, transform=transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ]))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
val_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)

# Train
model = Autoencoder()
device = torch.device('cuda:0')
model.to(device)

model_name = f'ae_{time.time()}'
print(model_name)

# Tensorboard
tb_writer = SummaryWriter('logs/' + model_name)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = ExponentialLR(optimizer, gamma=0.999)

# Loss function
loss_func = torch.nn.MSELoss()
best_loss = 10000
for epoch in range(500):
    print(f'Epoch: {epoch}')
    current_time = time.time()

    # Train
    model.train()
    loss_tr = []
    for step, (images_raw, _) in enumerate(train_loader):
        images_raw = images_raw.to(device)
        p = model(images_raw.float())
        batch_loss = loss_func(images_raw, p)
        loss_tr.append(batch_loss.detach().item())
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
    print(f"{epoch}, training_loss {np.mean(loss_tr)}, {time.time() - current_time} secs")
    current_time = time.time()

    # Validation
    model.eval()
    loss_ts = []
    for step, (images_raw, _) in enumerate(val_loader):
        images_raw = images_raw.to(device)
        p = model(images_raw.float())
        batch_loss = loss_func(images_raw, p)
        loss_ts.append(batch_loss.detach().cpu().numpy())
    print(f"{epoch}, validation_loss {np.mean(loss_ts)}, {time.time() - current_time} secs")
    scheduler.step()

    test_loss = np.mean(loss_ts)

    tb_writer.add_scalar("Test Loss", test_loss, epoch)
    tb_writer.add_scalar("Training Loss", np.mean(loss_tr), epoch)
    tb_writer.add_scalar("Learning Rate", scheduler.get_lr()[0], epoch)
    img_grid = torchvision.utils.make_grid(images_raw[:4].cpu().detach())
    tb_writer.add_image('orig_cifar_10_images', img_grid)
    img_grid = torchvision.utils.make_grid(p[:4].cpu().detach())
    tb_writer.add_image('recons_cifar_10_images', img_grid)
    tb_writer.flush()

    torch.save(model, "c10_ae_checkpoint/checkpoint.pth.tar")
    if test_loss < best_loss:
        torch.save(model, "c10_ae_checkpoint/model_best.pth.tar")
        best_loss = test_loss
