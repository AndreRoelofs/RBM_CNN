'''
Training script for Fashion-MNIST
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import copy
import random
# from sam import SAM
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import random_erasing.models.fashion as models
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
parser.add_argument('-d', '--dataset', default='fashionmnist', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0.0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+',
                    # default=[150, 225],
                    default=[100],
                    # default=[50, 75],
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
#                     help='path to latest checkpoint (default: none)')
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
parser.add_argument('--depth', type=int, default=28, help='Model depth.')
parser.add_argument('--widen-factor', type=int, default=10, help='Widen factor. 10')
# parser.add_argument('--depth', type=int, default=40, help='Model depth.')
# parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 10')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', default=0, type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

# Random Erasing
# parser.add_argument('--p', default=1.0, type=float, help='Random Erasing probability')
parser.add_argument('--sh', default=0.8, type=float, help='max erasing area')
parser.add_argument('--r1', default=0.7, type=float, help='aspect of erasing area')

parser.add_argument('--p', default=0.5, type=float, help='Random Erasing probability')
# parser.add_argument('--sh', default=0.4, type=float, help='max erasing area')
# parser.add_argument('--r1', default=0.3, type=float, help='aspect of erasing area')
#
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'fashionmnist'

# Use CUDA
use_cuda = torch.cuda.is_available()

# Random seed
# if args.manualSeed is None:
#     args.manualSeed = random.randint(1, 10000)
# random.seed(args.manualSeed)
# np.random.seed(args.manualSeed)
# torch.manual_seed(args.manualSeed)
# if use_cuda:
#     torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy


def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.GaussianBlur(3),
        # transforms.RandomRotation(1),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomErasing(probability=args.p, sh=args.sh, r1=args.r1, mean=[0.4914]),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    if args.dataset == 'fashionmnist':
        dataloader = datasets.FashionMNIST
        num_classes = 10

    indices = np.load('fashion_mnist_training_indices.npy')
    train_indices = indices[:50000]
    val_indices = indices[50000:]

    trainset = dataloader(root='../data', train=True, download=True, transform=transform_train)
    valset = dataloader(root='../data', train=True, download=True, transform=transform_test)
    testset = dataloader(root='../data', train=False, download=False, transform=transform_test)

    n_clusters = 80
    model_number = 4

    train_predictions = np.load(
        "../one_layered_wdn/train_clusters_Fashion_MNIST_old_rbm_cnn_data_normalized_quality_wide_levels_1_{}_{}.npy".format(
            model_number, n_clusters))
    # val_predictions = np.load(
    #     "../one_layered_wdn/val_clusters_Fashion_MNIST_old_val_rbm_cnn_data_normalized_quality_wide_levels_1_{}_{}.npy".format(
    #         model_number, n_clusters))
    test_predictions = np.load(
        "../one_layered_wdn/test_clusters_Fashion_MNIST_old_rbm_cnn_data_normalized_quality_wide_levels_1_{}_{}.npy".format(
            model_number, n_clusters))

    # train_predictions = np.load("../one_layered_wdn/1_level_train_clusters_80_Fashion_MNIST_rbm_fixed_7_val.npy")
    # val_predictions = np.load("../one_layered_wdn/1_level_val_clusters_80_Fashion_MNIST_rbm_fixed_7_val.npy")
    # test_predictions = np.load("../one_layered_wdn/1_level_test_clusters_80_Fashion_MNIST_rbm_fixed_7_val.npy")
    #
    # train_predictions = np.load("../one_layered_wdn/1_level_train_clusters_80_rbm_fixed_5.npy")
    # test_predictions = np.load("../one_layered_wdn/1_level_test_clusters_80_rbm_fixed_5.npy")

    title = 'fashionmnist-' + args.arch
    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
    test_correct_preds = []
    val_correct_preds = []
    best_acc = 0
    n_accuracy_decimals = 3
    for cluster_id in range(0, train_predictions.max() + 1):
    # for cluster_id in [13]:
        state['lr'] = args.lr

        # for cluster_id in range(0, 1):
        print("Current cluster ", cluster_id)
        train_cluster_idx = []
        val_cluster_idx = []

        for i in range(len(train_predictions)):
            cluster = train_predictions[i]
            if cluster != cluster_id:
                continue
            # train_cluster_idx.append(train_indices[i])
            #
            if i in train_indices:
                train_cluster_idx.append(i)
            else:
                val_cluster_idx.append(i)

            # if i in val_indices:
            #     val_cluster_idx.append(i)
            # train_cluster_idx.append(i)

        print("Train size: {}".format(len(train_cluster_idx)))

        random.shuffle(train_cluster_idx)

        trainloader = data.DataLoader(
            trainset,
            batch_size=min(args.train_batch, len(train_cluster_idx)),
            # batch_size=min(64, len(train_cluster_idx)),
            shuffle=False,
            num_workers=args.workers,
            # sampler=SubsetRandomSampler(train_cluster_idx),
            sampler=ImbalancedDatasetSampler(dataset=trainset, indices=train_cluster_idx),
        )
        #
        # for i in range(len(val_predictions)):
        #     cluster = val_predictions[i]
        #     if cluster != cluster_id:
        #         continue
        #
        #     val_cluster_idx.append(val_indices[i])
        print("Val size: {}".format(len(val_cluster_idx)))

        valloader = data.DataLoader(
            valset,
            batch_size=min(args.test_batch, len(val_cluster_idx)),
            shuffle=False,
            num_workers=args.workers,
            sampler=SubsetRandomSampler(val_cluster_idx),
            # sampler=ImbalancedDatasetSampler(dataset=valset, indices=val_cluster_idx),

        )

        test_cluster_idx = []
        for i in range(len(test_predictions)):
            cluster = test_predictions[i]
            if cluster != cluster_id:
                continue
            test_cluster_idx.append(i)

        print("Test size: {}".format(len(test_cluster_idx)))

        if len(test_cluster_idx) == 0:
            continue

        testloader = data.DataLoader(
            testset,
            batch_size=min(args.test_batch, len(test_cluster_idx)),
            shuffle=False,
            num_workers=args.workers,
            sampler=SubsetRandomSampler(test_cluster_idx)

        )
        print("Train batch: ", args.train_batch)

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
        #
        # fc_clone = copy.deepcopy(model.fc)
        #
        # for param in model.parameters():
        #     param.requires_grad = False
        #
        # model.fc = fc_clone
        # model.fc = nn.Linear(64*args.widen_factor, 10)
        # print(nn.Linear(64*args.widen_factor, 10).weight)
        # exit(1)

        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
        print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

        # Resume
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        # best_acc = checkpoint['best_acc']
        best_acc = 0
        # start_epoch = checkpoint['epoch']
        start_epoch = 0
        model.load_state_dict(checkpoint['state_dict'])
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = args.lr

        # logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer.load_state_dict(checkpoint['optimizer'])
        #
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
        # #
        # print(args.lr)

        train_loss, train_acc, _ = test(trainloader, model, criterion, 0, use_cuda)
        print("Original Train Accuracy: {} Loss: {}".format(train_acc, train_loss))
        val_loss, val_acc, _ = test(valloader, model, criterion, 0, use_cuda)
        print("Original Valid Accuracy: {} Loss: {}".format(val_acc, val_loss))
        test_loss, test_acc, _ = test(testloader, model, criterion, 0, use_cuda)
        print("Original Test Accuracy: {} Loss: {}".format(test_acc, test_loss))

        val_acc = np.around(val_acc, n_accuracy_decimals)

        # best_acc = 0
        # best_acc = test_acc
        best_acc = val_acc
        og_val_acc = val_acc
        og_test_acc = test_acc

        og_test_acc = np.around(og_test_acc, n_accuracy_decimals)
        og_val_acc = np.around(og_val_acc, n_accuracy_decimals)

        # max_val_correct = 0
        # Train and val
        for epoch in range(start_epoch, args.epochs):
            if best_acc >= 100:
                break
            if og_test_acc >= 100:
                break
            adjust_learning_rate(optimizer, epoch)

            # print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

            train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
            # print("Val classes:")
            val_loss, val_acc, _ = test(valloader, model, criterion, epoch, use_cuda)
            # print("Test classes:")
            test_loss, test_acc, _ = test(testloader, model, criterion, 0, use_cuda)
            print('Epoch: [%d | %d] LR: %f \nBest Accuracy: %.3f Valid Accuracy: %.3f Test Accuracy: %.3f' % (
                epoch + 1, args.epochs, state['lr'], best_acc, val_acc, test_acc))
            # print('Epoch: [%d | %d] LR: %f Best Accuracy: %f Valid Accuracy: %f' % (
            #     epoch + 1, args.epochs, state['lr'], best_acc, val_acc))

            if epoch <= 5:
                continue

            # exit(1)

            # append logger file
            logger.append([state['lr'], train_loss, val_loss, train_acc, val_acc])

            # save model
            # is_best = test_acc > best_acc + 0.00001
            # best_acc = max(test_acc, best_acc)
            #
            # is_best = val_acc > best_acc + 0.0001
            val_acc = np.around(val_acc, n_accuracy_decimals)
            is_best = val_acc >= best_acc
            best_acc = max(val_acc, best_acc)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': val_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

        # if best_acc <= og_test_acc + 0.0001:
        if best_acc <= og_val_acc:
            model = models.__dict__[args.arch](
                num_classes=num_classes,
                depth=args.depth,
                widen_factor=args.widen_factor,
                dropRate=args.drop,
            )
            model = torch.nn.DataParallel(model).cuda()
            checkpoint = torch.load('checkpoint/model_best_og.pth.tar')
        else:
            checkpoint = torch.load('checkpoint/model_best.pth.tar')

        model.load_state_dict(checkpoint['state_dict'])
        test_loss, test_acc, n_test_incorrect = test(testloader, model, criterion, 0, use_cuda)
        print("New Test Accuracy: {} Loss: {}".format(test_acc, test_loss))

        test_acc = np.around(test_acc, n_accuracy_decimals)


        # logger.close()
        # logger.plot()
        # savefig(os.path.join(args.checkpoint, 'log.eps'))

        print('Best acc:')
        print(best_acc)
        print("Og Val Acc Diff:")
        print(best_acc - og_val_acc)
        print("Og Test Acc Diff:")
        # print(best_acc - og_test_acc)
        print(test_acc - og_test_acc)
        # val_correct_preds.append(int((best_acc / 100) * len(val_cluster_idx)))
        test_correct_preds.append(len(test_cluster_idx) - n_test_incorrect)
        # test_correct_preds.append(max_test_correct)

    # best_accuracies.append(best_acc)
    # print("Total of val correct preds: {}".format(np.sum(val_correct_preds)))
    print("Total of test correct preds: {}".format(np.sum(test_correct_preds)))


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
                # incorrect_classes += targets[incorrect_indices].cpu().detach().numpy().flatten().tolist()
                incorrect_classes += pred.t()[0][incorrect_indices].cpu().detach().numpy().flatten().tolist()

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
    # incorrect_classes.sort()
    # if len(incorrect_classes) > 0:
    #     print("Incorrect classes: ", incorrect_classes)
    # print(top1.avg)
    return (losses.avg, top1.avg, len(incorrect_classes))


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
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
