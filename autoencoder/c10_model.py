import torch
from torch import nn

class Autoencoder(torch.nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 8, kernel_size=3, padding=1)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(8, 16, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(16, 32, kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 64, kernel_size=3, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1)

        self.lrelu = nn.LeakyReLU()
        # self.lrelu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)

    def encode(self, x):
        indices = []

        x = self.lrelu(self.conv1(x))
        x, indice = self.pool(x)
        indices.append(indice)

        x = self.lrelu(self.conv2(x))

        x = self.lrelu(self.conv3(x))
        x, indice = self.pool(x)
        indices.append(indice)

        x = self.lrelu(self.conv4(x))

        return x, indices

    def decode(self, x, indices):
        x = self.lrelu(self.deconv1(x))
        x = self.unpool(x, indices[1])

        x = self.lrelu(self.deconv2(x))

        x = self.lrelu(self.deconv3(x))
        x = self.unpool(x, indices[0])

        x = self.deconv4(x)

        return torch.sigmoid(x)

    def forward(self, x):
        x, indices = self.encode(x)
        x = self.decode(x, indices)
        return x

