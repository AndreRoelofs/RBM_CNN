import torch
from torch import nn
from torch.nn import functional as F


class FullyConnectedClassifier(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.device = torch.device("cuda")

        self.fc1 = nn.Linear(n_features, 50)
        # self.fc2 = nn.Linear(400, 200)
        # self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(50, 10)

        self.fc1_bn = nn.BatchNorm1d(50)
        # self.fc2_bn = nn.BatchNorm1d(200)
        # self.fc3_bn = nn.BatchNorm1d(100)
        self.fc4_bn = nn.BatchNorm1d(10)

        self.act = nn.SELU()
        self.to(self.device)

    def forward(self, x):
        x = self.fc1_bn(self.fc1(x))
        x = self.act(x)
        x = self.fc4(x)


        return F.log_softmax(x, dim=1)

    def loss_function(self, x, y):
        # return F.kl_div(x, y)
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
    clf.train()
    for epoch in range(10):
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
