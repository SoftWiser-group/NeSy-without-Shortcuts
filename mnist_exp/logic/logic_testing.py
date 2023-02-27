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
from logic_encoder import *
import models

import os
use_cuda = torch.cuda.is_available()

from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

parser = argparse.ArgumentParser(description='PyTorch CIFAR-100 Logic Training')
parser.add_argument('--seed', default=1, type=int, help='Random seed to use.')
parser.add_argument('--dataset', default='mnist', type=str, help='Data set.')
parser.add_argument('--exp_name', default='', type=str, help='Experiment name')
parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint')
parser.add_argument('--tol', type=float, default=1e-2, help='Tolerance for constraints')
args = parser.parse_args()
sys.path.append('../')
from config import *

eps = 0.1

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
best_acc = 0
best_model = None
# Data Uplaod
print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([
        transforms.Resize([32,32]),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)), 
    ])

transform_rotate = transforms.Compose([
        transforms.Resize([32,32]),
        transforms.RandomRotation(degrees=[180,180]),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)), 
    ])

transform_test = transforms.Compose([
        transforms.Resize([32,32]),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)), 
    ])

print("| Preparing dataset...")
sys.stdout.write("| ")
if args.dataset == 'mnist':
    MNIST = dataset_with_indices(torchvision.datasets.MNIST)
    trainset = MNIST(root='../../data/mnist', train=True, download=True, transform=transform_train)
    rotateset = MNIST(root='../../data/mnist', train=False, download=True, transform=transform_rotate)
    testset = MNIST(root='../../data/mnist', train=False, download=False, transform=transform_test)
elif args.dataset == 'usps':
    USPS = dataset_with_indices(torchvision.datasets.USPS)
    trainset = USPS(root='../../data/usps', train=True, download=True, transform=transform_train)
    rotateset = USPS(root='../../data/usps', train=False, download=True, transform=transform_rotate)
    testset = USPS(root='../../data/usps', train=False, download=False, transform=transform_test)
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

def cons_sat(probs_u, probs_r):
    batch_size = probs_u.shape[0]
    # or_res1 = BatchOr([LE(probs_u[:,6]-probs_u[:,i], -0.01) for i in [0,1,2,3,4,5,7,8,9]], batch_size)
    # l1 = [GE(probs_u[:,6]-probs_u[:,i], 0.01) for i in [0,1,2,3,4,5,7,8,9]]
    # l2 = [GE(probs_r[:,9]-probs_r[:,i], 0.01) for i in [0,1,2,3,4,5,6,7,8]]
    # and_res2 = BatchAnd(l1 + l2, batch_size)
    # 
    or_res1 = BatchOr([LE(probs_r[:,9]-probs_r[:,i], -eps) for i in [0,1,2,3,4,5,6,7,8]], batch_size)
    and_res2 = BatchAnd([GE(probs_u[:,6]-probs_u[:,i], eps) for i in [0,1,2,3,4,5,7,8,9]], batch_size)
    #
    # or_res1 = BatchAnd([LE(probs_r[:,9], 0.1), LE(probs_u[:,6], 0.1)],batch_size)
    # and_res2 = GE(probs_u[:,6], 0.9)

    ans_sat1 = or_res1.satisfy(args.tol)
    ans_sat2 = and_res2.satisfy(args.tol)

    all_res = BatchOr([or_res1, and_res2], batch_size)
    ans_sat = all_res.satisfy(args.tol)
    return ans_sat1, ans_sat2, ans_sat

net.eval()
### test results
criterion = nn.CrossEntropyLoss()
test_loss = 0
correct = 0
constraint_correct1 = 0
constraint_correct2 = 0
constraint_correct = 0
constraint_num = 0
total = 0
for batch_idx, (inputs, targets, index) in enumerate(testloader):
    if use_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
    inputs, targets = Variable(inputs), Variable(targets)
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    probs = softmax(outputs)

    ind = np.where(targets.cpu().detach().numpy() == 6)[0]
    if len(ind) != 0:
        probs_u = probs[ind,:]
        index = index[ind]
        inputs_r = [rotateset[i][0] for i in index]
        inputs_r = torch.cat(inputs_r, dim=0).unsqueeze(1)
        outputs_r = net(inputs_r)
        probs_r = softmax(outputs_r)
        ans_sat1, ans_sat2, ans_sat = cons_sat(probs_u, probs_r)
        constraint_correct1 += ans_sat1.sum()
        constraint_correct2 += ans_sat2.sum()
        constraint_correct += ans_sat.sum()
        constraint_num += len(ind)

    test_loss += loss.item()
    _, predicted = torch.max(outputs.data, 1)
    total += targets.size(0)
    correct += predicted.eq(targets.data).cpu().sum()

# Save checkpoint when best model
acc = 100.*float(correct)/total
cons_acc1 = 100.*float(constraint_correct1)/constraint_num
cons_acc2 = 100.*float(constraint_correct2)/constraint_num
cons_acc = 100.*float(constraint_correct)/constraint_num
total_acc = (acc) 
print("\n| Test results #\t\t\tLoss: %.4f Acc@1: %.2f%% Cons_Acc1+2/Cons_Acc: %.2f%%+%.2f%%/%.2f%%" %(loss.item(), acc, cons_acc1, cons_acc2, cons_acc))

