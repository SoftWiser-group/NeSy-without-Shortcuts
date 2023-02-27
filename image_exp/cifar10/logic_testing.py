from __future__ import print_function

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import pickle
import math

from tqdm import tqdm
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import time

import sys
sys.path.append('../../')
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from logic_encoder import *
import models

import os
use_cuda = torch.cuda.is_available()

from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Logic Training')
parser.add_argument('--seed', default=1, type=int, help='Random seed to use.')
parser.add_argument('--dataset', default='cifar10', type=str, help='Data set.')
parser.add_argument('--net_type', default='resnet50', type=str, help='Model')
parser.add_argument('--num_labeled', default=100, type=int, help='Number of labeled examples (per class!).')
parser.add_argument('--exp_name', default='', type=str, help='Experiment name')
parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint')
parser.add_argument('--tol', type=float, default=1e-2, help='Tolerance for constraints')
args = parser.parse_args()
sys.path.append('../')
from config import *

if args.dataset == 'cifar10':
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[args.dataset], std[args.dataset]),
    ]) # meanstd transformation
elif args.dataset == 'stl10':
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[args.dataset], std[args.dataset]),
    ]) # meanstd transformation

if args.dataset == 'stl10':
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean[args.dataset], std[args.dataset]),
    ])
elif args.dataset == 'cifar10':
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean[args.dataset], std[args.dataset]),
    ])

if args.dataset == 'cifar10':
    torchvision.datasets.CIFAR10(root='../../data/cifar10', train=True, download=True)
    trainset = torchvision.datasets.CIFAR10(root='../../data/cifar10', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='../../data/cifar10', train=False, download=False, transform=transform_test)
elif args.dataset == 'stl10':
    torchvision.datasets.STL10(root='../../data/STL10', split='train', download=True)
    trainset = torchvision.datasets.STL10(root='../../data/STL10', split='train', download=True, transform=transform_train)
    testset = torchvision.datasets.STL10(root='../../data/STL10', split='test', download=False, transform=transform_test)

C1 = [0, 1, 8, 9] # 
C2 = [2, 3, 4, 5, 6, 7] # animals
C = [C1, C2]

group = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]


# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
# Data Uplaod
print('\n[Phase 1] : Data Preparation')

print("| Preparing {} dataset...".format(args.dataset))
sys.stdout.write("| ")
num_classes = 10

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
print('\n[Test Phase] : Model setup')
assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
# _, file_name = getNetwork(args)
checkpoint = torch.load('./checkpoint/' + args.resume_from + '.t7')
net = checkpoint['net']
print('| resume from the net type [' + args.resume_from + ']...')


if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

def cons_sat(probs):
    batch_size = probs.shape[0]
    cons = []
    or_sat = []
    for i in range(2):
        gsum = 0
        for j in C[i]:
            gsum += probs[:,j]
        or1 = EQ(gsum, 1.0)
        or0 = EQ(gsum, 0.0)
        or_res = BatchOr([or1, or0], batch_size)
        cons.append(or_res)
        or_sat.append(torch.stack((or1.satisfy(args.tol), or0.satisfy(args.tol)), 0))
    and_res = BatchAnd(cons, batch_size)  
    ans_sat = and_res.satisfy(args.tol)
    return ans_sat, or_sat

net.eval()
### training results
test_loss = 0
correct = 0
constraint_correct = 0
total = 0

conf_mat = np.zeros((100, 100))
group_ok = 0

np.set_printoptions(threshold=np.inf)
softmax = torch.nn.Softmax(dim=1)
for batch_idx, (inputs, targets) in enumerate(trainloader):
    if use_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
    inputs, targets = Variable(inputs), Variable(targets)
    outputs = net(inputs)
    
    probs = softmax(outputs)
    ans_sat, or_sat = cons_sat(probs)
    constraint_correct += ans_sat.sum()


    _, predicted = torch.max(outputs.data, 1)
    total += targets.size(0)
    correct += predicted.eq(targets.data).cpu().sum()

    conf_mat += confusion_matrix(targets.data.cpu().numpy(), predicted.cpu().numpy(), labels=np.arange(100))

    n_batch = predicted.size()[0]
    for i in range(n_batch):
        if group[predicted.cpu()[i]] == group[targets.cpu().data[i]]:
            group_ok += 1

#rint('Confusion matrix:')
#print(conf_mat)

    
acc = 100.0*float(correct)/total
c_acc = 100.0*float(constraint_correct)/total
group_acc = 100.0*float(group_ok)/total
print("| Train Result\tAcc@1: %.2f%%" %(acc))
print("| Train Result\tCAcc: %.2f%%" %(c_acc))
print("| Train Result\tGroupAcc: %.2f%%" %(group_acc))

### testing results
test_loss = 0
correct = 0
constraint_correct = 0
total = 0

conf_mat = np.zeros((100, 100))
group_ok = 0


np.set_printoptions(threshold=np.inf)
for batch_idx, (inputs, targets) in enumerate(testloader):
    if use_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
    inputs, targets = Variable(inputs), Variable(targets)
    outputs = net(inputs)
    
    probs = softmax(outputs)
    ans_sat, or_sat = cons_sat(probs)
    constraint_correct += ans_sat.sum()

    _, predicted = torch.max(outputs.data, 1)
    total += targets.size(0)
    correct += predicted.eq(targets.data).cpu().sum()

    conf_mat += confusion_matrix(targets.data.cpu().numpy(), predicted.cpu().numpy(), labels=np.arange(100))

    n_batch = predicted.size()[0]
    for i in range(n_batch):
        if group[predicted.cpu()[i]] == group[targets.cpu().data[i]]:
            group_ok += 1
          

#rint('Confusion matrix:')
#print(conf_mat)
    
acc = 100.0*float(correct)/total
c_acc = 100.0*float(constraint_correct)/total
group_acc = 100.0*float(group_ok)/total
print("| Test Result\tAcc@1: %.2f%%" %(acc))
print("| Test Result\tCAcc: %.2f%%" %(c_acc))
print("| Test Result\tGroupAcc: %.2f%%" %(group_acc))
