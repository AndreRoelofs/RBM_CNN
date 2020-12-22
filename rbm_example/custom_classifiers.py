import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F


class ClassifierV1(nn.Module): #similar to a basic SVM
    def __init__(self):
        super().__init__()
        self.fully_connected = nn.Linear(4, 1)

    def forward(self, x):
        fwd = self.fully_connected(x)
        return fwd

class ClassifierV2(nn.Module): #MLP
    def __init__(self):
        super(ClassifierV2, self).__init__()
        # number of hidden nodes in each layer (512)
        hidden_1 = 512
        hidden_2 = 512
        # linear layer (4 -> hidden_1)
        self.fc1 = nn.Linear(4, 512)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(512, 512)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(512, 10)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.droput = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 4)
        # add hidden layer, with relu activation function
        x = F.selu(self.fc1(x))
        # add dropout layer
        x = self.droput(x)
        # add hidden layer, with relu activation function
        x = F.selu(self.fc2(x))
        # add dropout layer
        x = self.droput(x)
        # add output layer
        x = self.fc3(x)
        return x