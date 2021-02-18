'''
Training script for Fashion-MNIST
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
from sam import SAM
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import random_erasing.models.cifar as models
from torch.utils.data.sampler import SubsetRandomSampler
import random_erasing.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from random_erasing.utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from ImbalancedDatasetSampler import ImbalancedDatasetSampler

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Fashion-MNIST Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=1, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0.0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+',
                    # default=[150, 225],
                    # default=[20, 40],
                    default=[100],
                    # default=[],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=10.0, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
# parser.add_argument('--resume', default='checkpoint/checkpoint.pth.tar', type=str, metavar='PATH',
#                     help='path to latest ccheckpoint (default: none)')
# parser.add_argument('--resume', default='checkpoint/model_best.pth.tar', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
parser.add_argument('--resume', default='checkpoint/model_best_og.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='wrn',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet20)')
# parser.add_argument('--depth', type=int, default=28, help='Model depth.')
# parser.add_argument('--widen-factor', type=int, default=10, help='Widen factor. 10')
parser.add_argument('--depth', type=int, default=28, help='Model depth.')
parser.add_argument('--widen-factor', type=int, default=10, help='Widen factor. 10')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', default=0, type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

# Random Erasing
parser.add_argument('--p', default=1.0, type=float, help='Random Erasing probability')
parser.add_argument('--sh', default=0.8, type=float, help='max erasing area')
parser.add_argument('--r1', default=0.7, type=float, help='aspect of erasing area')
#
# parser.add_argument('--p', default=0.5, type=float, help='Random Erasing probability')
# parser.add_argument('--sh', default=0.4, type=float, help='max erasing area')
# parser.add_argument('--r1', default=0.3, type=float, help='aspect of erasing area')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy


def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last fm_ae_checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.GaussianBlur(3),
        # transforms.RandomRotation(1),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(probability=args.p, sh=args.sh, r1=args.r1, ),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)

    # trainset.data = trainset.data[:10000]
    # trainset.targets = trainset.targets[:10000]
    #
    # testset.data = testset.data[:1000]
    # testset.targets = testset.targets[:1000]

    # train_predictions = np.load("../one_layered_wdn/1_level_train_clusters_40_large_rbm_fixed_max_rbm.npy")
    # test_predictions = np.load("../one_layered_wdn/1_level_test_clusters_40_large_rbm_fixed_max_rbm.npy")

    train_predictions = np.load("../one_layered_wdn/1_level_train_clusters_80_CIFAR_10_large_rbm_fixed_3.npy")
    test_predictions = np.load("../one_layered_wdn/1_level_test_clusters_80_CIFAR_10_large_rbm_fixed_3.npy")

    # train_predictions = np.load("../one_layered_wdn/1_level_train_clusters_2_large_rbm_fixed_3.npy")
    # test_predictions = np.load("../one_layered_wdn/1_level_test_clusters_2_large_rbm_fixed_3.npy")
    #
    # train_predictions = np.load("../autoencoder/cifar_10_ae_512_train_clusters_80.npy")
    # test_predictions = np.load("../autoencoder/cifar_10_ae_512_test_clusters_80.npy")

    # train_predictions = np.load("../one_layered_wdn/1_level_train_clusters_80_CIFAR_10_rbm_fixed_5.npy")
    # test_predictions = np.load("../one_layered_wdn/1_level_test_clusters_80_CIFAR_10_rbm_fixed_5.npy")

    title = 'cifar-10-' + args.arch
    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
    correct_preds = []
    best_acc = 0
    for cluster_id in range(80):
        # for cluster_id in [1]:
        state['lr'] = args.lr

        # for cluster_id in range(0, 1):
        print("Current cluster ", cluster_id)
        train_cluster_idx = []
        for i in range(len(train_predictions)):
            cluster = train_predictions[i]
            if cluster != cluster_id:
                continue
            train_cluster_idx.append(i)

        print("Train size: {}".format(len(train_cluster_idx)))

        trainloader = data.DataLoader(
            trainset,
            batch_size=min(args.train_batch, len(train_cluster_idx)),
            # batch_size=min(64, len(train_cluster_idx)),
            shuffle=False,
            num_workers=args.workers,
            # sampler=SubsetRandomSampler(train_cluster_idx),
            sampler=ImbalancedDatasetSampler(dataset=trainset, indices=train_cluster_idx),
        )

        print("Train batch: ", args.train_batch)

        test_cluster_idx = []
        for i in range(len(test_predictions)):
            cluster = test_predictions[i]
            if cluster != cluster_id:
                continue
            test_cluster_idx.append(i)

        print("Test size: {}".format(len(test_cluster_idx)))

        testloader = data.DataLoader(
            testset,
            batch_size=min(args.test_batch, len(test_cluster_idx)),
            shuffle=False,
            num_workers=args.workers,
            sampler=SubsetRandomSampler(test_cluster_idx)

        )

        # Model
        print("==> creating model '{}'".format(args.arch))
        if args.arch.startswith('wrn'):
            model = models.__dict__[args.arch](
                num_classes=num_classes,
                depth=args.depth,
                widen_factor=args.widen_factor,
                dropRate=args.drop,
            )
        elif args.arch.endswith('resnet'):
            model = models.__dict__[args.arch](
                num_classes=num_classes,
                depth=args.depth,
            )

        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
        print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(model.parameters(), lr=1e-4)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        # base_optimizer = optim.SGD
        # optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # Resume
        # Load fm_ae_checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        # best_acc = fm_ae_checkpoint['best_acc']
        best_acc = 0
        # start_epoch = fm_ae_checkpoint['epoch']
        start_epoch = 0
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(fm_ae_checkpoint['optimizer'])
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = args.lr

        # logger = Logger(os.path.join(args.fm_ae_checkpoint, 'log.txt'), title=title, resume=True)

        train_loss, train_acc = test(trainloader, model, criterion, 0, use_cuda)
        print("Original Train Accuracy: {} Loss: {}".format(train_acc, train_loss))
        test_loss, test_acc = test(testloader, model, criterion, 0, use_cuda)
        print("Original Test Accuracy: {} Loss: {}".format(test_acc, test_loss))
        best_acc = test_acc
        og_acc = test_acc

        # Train and val
        for epoch in range(start_epoch, args.epochs):
            if best_acc == 100:
                break
            adjust_learning_rate(optimizer, epoch)

            # print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

            train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
            test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)
            print('Epoch: [%d | %d] LR: %f Best Accuracy: %f Test Accuracy: %f' % (
            epoch + 1, args.epochs, state['lr'], best_acc, test_acc))

            # append logger file
            logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

            # save model
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

        # logger.close()
        # logger.plot()
        # savefig(os.path.join(args.fm_ae_checkpoint, 'log.eps'))

        print('Best acc:')
        print(best_acc)
        print("Og Acc Diff:")
        print(best_acc - og_acc)
        correct_preds.append(int((best_acc / 100) * len(test_cluster_idx)))
        # best_accuracies.append(best_acc)
    print("Total of correct preds: {}".format(np.sum(correct_preds)))


def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)

        # Normal
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #
        # # SAM
        # criterion(outputs, targets).backward()
        # first forward-backward pass
        # optimizer.first_step(zero_grad=True)
        # outputs = model(inputs)
        # second forward-backward pass
        # loss = criterion(outputs, targets)
        # loss.backward()
        # optimizer.second_step(zero_grad=True)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    incorrect_classes = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, pred = outputs.topk(1, 1, True, True)
            incorrect_indices = (pred.t()[0] != targets).nonzero().t()[0]
            if incorrect_indices.shape[0] > 0:
                incorrect_classes += targets[incorrect_indices].cpu().detach().numpy().flatten().tolist()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(testloader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
    bar.finish()
    incorrect_classes.sort()
    if len(incorrect_classes) > 0:
        print("Incorrect classes: ", incorrect_classes)
    # print(top1.avg)
    return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint='fm_ae_checkpoint', filename='fm_ae_checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    # global args
    # if epoch in args.schedule:
    #     args.train_batch *= args.gamma
    #     args.train_batch = int(args.train_batch)
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == '__main__':
    main()
