'''
Training script for Fashion-MNIST
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
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
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import copy
from random_erasing.utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from ImbalancedDatasetSampler import ImbalancedDatasetSampler
import wandb
from sys import exit

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Fashion-MNIST Training')
# Datasets
parser.add_argument('-d', '--dataset', default='fashionmnist', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0.0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+',
                    # default=[150, 225],
                    # default=[20, 40],
                    # default=[100],
                    default=[],
                    help='Decre11ase learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
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
#
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
#
# # Random seed
# if args.manualSeed is None:
#     args.manualSeed = random.randint(1, 10000)
#
# print('Seed: {}'.format(args.manualSeed))
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

    trainset = dataloader(root='../data', train=True, download=True, transform=transform_train)
    testset = dataloader(root='../data', train=False, download=False, transform=transform_test)

    n_clusters = 160

    for model_number in range(7, 8):
        # wandb.init(project="Clusters_Fashion_MNIST_old_rbm_cnn_extra_training_levels_1_clusters_{}".format(n_clusters),
        #            reinit=True)
        train_predictions = np.load(
            "../one_layered_wdn/train_clusters_Fashion_MNIST_old_rbm_cnn_data_normalized_quality_wide_levels_1_{}_{}_compressed.npy".format(model_number, n_clusters))
        test_predictions = np.load(
            "../one_layered_wdn/test_clusters_Fashion_MNIST_old_rbm_cnn_data_normalized_quality_wide_levels_1_{}_{}_compressed.npy".format(model_number, n_clusters))

        title = 'fashionmnist-' + args.arch
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
        correct_preds = []
        best_acc = 0
        # for cluster_id in range(0, 12):
        for cluster_id in range(0, train_predictions.max() + 1):
        # for cluster_id in [6]:
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
                sampler=SubsetRandomSampler(train_cluster_idx),
                # sampler=ImbalancedDatasetSampler(dataset=trainset, indices=train_cluster_idx),
            )

            print("Train batch: ", args.train_batch)

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

            # for p in model.fc.parameters():
            #     print(p)
            # print(model.fc.parameters())

            # for p in model.fc.parameters():
            #     print(p)
            # exit(1)

            model = torch.nn.DataParallel(model).cuda()
            cudnn.benchmark = True
            print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

            # Resume
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
            args.checkpoint = os.path.dirname(args.resume)
            checkpoint = torch.load(args.resume)
            best_acc = 0
            start_epoch = 0
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # print(checkpoint['optimizer'])

            # for param in model.module.parameters():
            #     param.requires_grad = False
            #
            # for param in model.module.fc.parameters():
            #     param.requires_grad = True

            # model.module.fc = fc_clone
            # model.module.fc = nn.Linear(64 * args.widen_factor, 10)
            # model.module.fc.cuda()

            train_loss, train_acc, _ = test(trainloader, model, criterion, 0, use_cuda)
            print("Original Train Accuracy: {} Loss: {}".format(train_acc, train_loss))
            test_loss, test_acc, og_incorrect = test(testloader, model, criterion, 0, use_cuda)
            print("Original Test Accuracy: {} Loss: {}".format(test_acc, test_loss))
            best_acc = test_acc
            og_acc = test_acc
            min_incorrect = og_incorrect

            # exit(1)

            # Train and val
            for epoch in range(start_epoch, args.epochs):
                if best_acc >= 100:
                    break
                adjust_learning_rate(optimizer, epoch)

                train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
                test_loss, test_acc, incorrect_classes = test(testloader, model, criterion, epoch, use_cuda)
                min_incorrect = min(min_incorrect, incorrect_classes)
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

            print('Best acc:')
            print(best_acc)
            print("Og Acc Diff:")
            print(best_acc - og_acc)
            correct_preds.append(len(test_cluster_idx) - min_incorrect)
        print("Total of correct preds: {}".format(np.sum(correct_preds)))
        wandb.log({"max_accuracy": np.sum(correct_preds), 'improvement': np.sum(correct_preds)-9629})


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

        # print(inputs.shape)
        #
        # plt.imshow(inputs[0][0].detach().cpu().numpy(), cmap='gray')
        # plt.savefig('test.png')
        #
        # first_layer_output = model.module.conv1(inputs)[0]
        #
        # fig = plt.figure(figsize=(4., 4.))
        # grid = ImageGrid(fig, 111,  # similar to subplot(111)
        #                  nrows_ncols=(4, 4),  # creates 2x2 grid of axes
        #                  axes_pad=0.1,  # pad between axes in inch.
        #                  )
        #
        # for ax, im in zip(grid, first_layer_output):
        #     # Iterating over the grid returns the Axes.
        #     im = im.detach().cpu().numpy()
        #     ax.imshow(im, cmap='gray')
        #     ax.axis('off')
        #
        # fig.suptitle('Trained Model: First layer output', fontsize=12)
        #
        # plt.savefig('test_conv.png')
        #
        # print(first_layer_output.shape)
        # exit(1)

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
    incorrect_classes.sort()
    if len(incorrect_classes) > 0:
        print("Incorrect classes: ", incorrect_classes)
    # print(top1.avg)
    # print(top5.avg)
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
