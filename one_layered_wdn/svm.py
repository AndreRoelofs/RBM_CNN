import os
import time
import argparse
import pickle
import gc
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, gradcheck

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


class Net(nn.Module):
    def __init__(self, n_feature, n_class):
        super(Net, self).__init__()
        self.fc = nn.Linear(n_feature, n_class)
        torch.nn.init.kaiming_uniform_(self.fc.weight)
        torch.nn.init.constant_(self.fc.bias, 0.1)

    def forward(self, x):
        output = self.fc(x)
        return output


class multiClassHingeLoss(nn.Module):
    def __init__(self, p=1, margin=1, weight=None, size_average=True):
        super(multiClassHingeLoss, self).__init__()
        self.p = p
        self.margin = margin
        self.weight = weight  # weight for each class, size=n_class, variable containing FloatTensor,cuda,reqiures_grad=False
        self.size_average = size_average

    def forward(self, output, y):  # output: batchsize*n_class
        # print(output.requires_grad)
        # print(y.requires_grad)
        output_y = output[torch.arange(0, y.size()[0]).long().cuda(), y.data.cuda()].view(-1, 1)  # view for transpose
        # margin - output[y] + output[i]
        loss = output - output_y + self.margin  # contains i=y
        # remove i=y items
        loss[torch.arange(0, y.size()[0]).long().cuda(), y.data.long().cuda()] = 0
        # max(0,_)
        loss[loss < 0] = 0
        # ^p
        if (self.p != 1):
            loss = torch.pow(loss, self.p)
        # add weight
        if (self.weight is not None):
            loss = loss * self.weight
        # sum up
        loss = torch.sum(loss)
        if (self.size_average):
            loss /= output.size()[0]  # output.size()[0]
        return loss


def train(epoch, model, optimizer, loss, trn_loader):
    model.train()
    training_loss = 0
    training_f1 = 0
    for batch_idx, (data, target) in enumerate(trn_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        target.data = target.data.long()
        tloss = loss(output, target)
        training_loss += tloss.item()
        tloss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        training_f1 += f1_score(target.data.cpu().numpy(), pred.cpu().numpy(), labels=np.arange(10).tolist(),
                                average='macro')
    if (epoch + 1) % 100 == 0:
        print('Epoch: {}'.format(epoch))
        print('Training set avg loss: {:.4f}'.format(training_loss / len(trn_loader)))
        print('Training set avg micro-f1: {:.4f}'.format(training_f1 / len(trn_loader)))


def test(epoch, model, tst_loader, y_test_list):
    model.eval()
    test_loss = 0
    preds = list()
    with torch.no_grad():
        for data, target in tst_loader:
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target.long(), size_average=False).item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            preds += pred.cpu().numpy().tolist()
    test_loss /= len(tst_loader.dataset)
    conf_mat = confusion_matrix(y_test_list, preds)
    precision, recall, f1, sup = precision_recall_fscore_support(y_test_list, preds, average='macro', zero_division=0)
    accuracy = accuracy_score(y_test_list, preds)
    if (epoch + 1) % 25 == 0:
        print('Test set avg loss: {:.4f}'.format(test_loss))
        print('conf_mat:\n', conf_mat)
        print('Precison:{:.4f}\nRecall:{:.4f}\nf1:{:.4f}\nAccuracy:{:.4f}'.format(precision, recall, f1, accuracy))
    return conf_mat, precision, recall, f1, accuracy
