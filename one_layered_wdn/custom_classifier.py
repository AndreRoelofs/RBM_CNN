import torch
from torch import nn
from torch.nn import functional as F


class FashionCNN(nn.Module):

    def __init__(self):
        super(FashionCNN, self).__init__()
        # self.device = torch.device("cpu")
        self.device = torch.device("cuda")

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)

        self.act = nn.SELU()
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        nn.init.xavier_normal_(self.conv1.weight, 20.0)
        nn.init.xavier_normal_(self.conv2.weight, 20.0)
        nn.init.xavier_normal_(self.conv8.weight, 20.0)

        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv8_bn = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(in_features=16*(14**2), out_features=200)
        self.drop = nn.Dropout2d(0.25)
        # self.fc2 = nn.Linear(in_features=200, out_features=120)
        self.fc3 = nn.Linear(in_features=200, out_features=10)

        self.to(self.device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.act(x)
        x = self.max_pool(x)
        #
        # x = self.conv2(x)
        # x = self.conv2_bn(x)
        # x = self.act(x)
        # x = self.max_pool(x)
        #
        # x = self.conv8(x)
        # x = self.conv8_bn(x)
        # x = self.act(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.drop(x)
        # x = self.fc2(x)
        # x = self.drop(x)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)

    def loss_function(self, x, y):
        return F.cross_entropy(x, y)
        # return F.nll_loss(x, y)


class FullyConnectedClassifier(nn.Module):
    def __init__(self, n_features, n_targets):
        super().__init__()
        self.device = torch.device("cuda")

        self.fc1 = nn.Linear(n_features, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, n_targets)

        self.fc1_bn = nn.BatchNorm1d(400)
        self.fc2_bn = nn.BatchNorm1d(200)
        self.fc3_bn = nn.BatchNorm1d(100)

        self.act = nn.SELU()
        self.to(self.device)

    def forward(self, x):
        # x = self.fc1_bn(self.fc1(x))
        # x = self.act(x)
        # x = self.fc2_bn(self.fc2(x))
        # x = self.act(x)
        # x = self.fc3_bn(self.fc3(x))
        # x = self.act(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = self.act(x)

        x = self.fc4(x)

        return F.log_softmax(x, dim=1)

    def loss_function(self, x, y):
        return F.nll_loss(x, y)


def predict_classifier(clf, test_dataset_loader):
    print("Making predictions")
    test_loss = 0
    correct = 0
    clf.eval()
    for batch_idx, (data, target) in enumerate(test_dataset_loader):
        data = data.to(clf.device)
        target = target.to(clf.device)
        out = clf(data)
        test_loss += clf.loss_function(out, target.long()).item()
        pred = out.data.max(1)[1]
        correct += pred.eq(target.data).sum()
    test_loss /= len(test_dataset_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_dataset_loader.dataset),
        100. * correct / len(test_dataset_loader.dataset)))


def train_classifier(clf, optimizer, train_dataset_loader, test_dataset_loader):
    for epoch in range(10):
        clf.train()
        for batch_idx, (data, target) in enumerate(train_dataset_loader):
            data = data.to(clf.device)
            target = target.to(clf.device)
            optimizer.zero_grad()
            out = clf(data)
            loss = clf.loss_function(out, target.long())
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_dataset_loader.dataset),
                           100.0 * batch_idx / len(train_dataset_loader), loss.item()))
        predict_classifier(clf, test_dataset_loader)
