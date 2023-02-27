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
os.environ['CUDA_VISIBLE_DEVICES'] = '8'
use_cuda = torch.cuda.is_available()

from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

parser = argparse.ArgumentParser(description='PyTorch CIFAR-100 Logic Training')
parser.add_argument('--seed', default=1, type=int, help='Random seed to use.')
parser.add_argument('--dataset', default='cifar100', type=str, help='Data set.')
parser.add_argument('--net_type', default='resnet50', type=str, help='Model')
parser.add_argument('--num_labeled', default=100, type=int, help='Number of labeled examples (per class!).')
parser.add_argument('--exp_name', default='', type=str, help='Experiment name')
parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint')
parser.add_argument('--tol', type=float, default=1e-2, help='Tolerance for constraints')
args = parser.parse_args()
sys.path.append('../')
from config import *
torchvision.datasets.CIFAR100(root='../../data/cifar100', train=True, download=True)
meta = pickle.load(open('../../data/cifar100/cifar-100-python/meta', 'rb'))
coarse = meta['coarse_label_names']
fine = meta['fine_label_names']

label_idx = {label:i for i, label in enumerate(fine)}
group_idx = {label:i for i, label in enumerate(coarse)}
g = {}
group = [0 for i in range(100)]
pairs = []

print(group_idx)

with open('groups.txt') as f:
    for line in f:
        tokens = line[:-1].split('\t')
        large_group = tokens[0]
        tokens[1] = tokens[1].replace(',', '').strip()
        labels = tokens[1].split(' ')
        assert len(labels) == 5, labels
        
        for label in labels:
            assert label in fine, label
            group[label_idx[label]] = group_idx[large_group]

        g[group_idx[large_group]] = [label_idx[label] for label in labels]
        
        for x in labels:
            for y in labels:
                if x != y:
                    pairs.append((label_idx[x], label_idx[y]))

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
# Data Uplaod
print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean[args.dataset], std[args.dataset]),
]) # meanstd transformation

transform_test = transforms.Compose([
    # transforms.RandomAffine(10, (0.3, 0.3), fill=(0,0,0)),
    transforms.ToTensor(),
    transforms.Normalize(mean[args.dataset], std[args.dataset]),
])

print("| Preparing CIFAR-100 dataset...")
sys.stdout.write("| ")
trainset = torchvision.datasets.CIFAR100(root='../../data/cifar100', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR100(root='../../data/cifar100', train=False, download=False, transform=transform_test)
num_classes = 100

# def getNetwork(args):
#     if args.net_type == 'resnet50':
#         net = models.ResNet50(100)
#         file_name = 'resnet50'
#     elif args.net_type == 'resnet18':
#         net = models.ResNet18(100)
#         file_name = 'resnet18'        
#     elif args.net_type == 'vgg16':
#         net = models.vgg16(100)
#         file_name = 'vgg16'
#     elif args.net_type == 'densenet100':
#         net = models.DenseNet100(100)
#         file_name = 'densenet100'
#     else:
#         assert False
#     file_name += '_' + str(args.seed) + '_' + args.exp_name
#     return net, file_name

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
print('\n[Test Phase] : Model setup')
assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
# _, file_name = getNetwork(args)
checkpoint = torch.load('./checkpoint/' + args.resume_from + '.t7')
net = checkpoint['net']
tau_and = checkpoint['tau_and']
tau_or = checkpoint['tau_or']
print('| resume from the net type [' + args.resume_from + ']...')

print('tau_and & tau_or: ', tau_and.shape, tau_or.shape)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

def cons_sat(probs):
    batch_size = probs.shape[0]
    cons = []
    or_sat = []
    for i in range(20):
        gsum = 0
        for j in g[i]:
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
sub_cons_correct = torch.zeros(tau_or.shape[1])
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
    # compute the satisfaction of sub expression
    or_sat = torch.cat(or_sat, 0)
    sub_cons_correct += torch.sum(or_sat,1).cpu().detach().numpy()


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

# pearson correlation of training set
tmp_or = tau_or.clone()
for i in range(20):
    tmp_or[:, 2*i:2*(i+1)] = softmax(tau_or[:, 2*i:2*(i+1)])
import scipy
print('pearson correlation of tau on training set: ', scipy.stats.pearsonr(sub_cons_correct, torch.sum(tmp_or, 0).cpu().detach().numpy()))
    
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
    # compute the satisfaction of sub expression
    or_sat = torch.cat(or_sat, 0)
    sub_cons_correct += torch.sum(or_sat,1).cpu().detach().numpy()


    _, predicted = torch.max(outputs.data, 1)
    total += targets.size(0)
    correct += predicted.eq(targets.data).cpu().sum()

    conf_mat += confusion_matrix(targets.data.cpu().numpy(), predicted.cpu().numpy(), labels=np.arange(100))

    n_batch = predicted.size()[0]
    for i in range(n_batch):
        if group[predicted.cpu()[i]] == group[targets.cpu().data[i]]:
            group_ok += 1

tmp_or = tau_or.clone()
for i in range(20):
    tmp_or[:, 2*i:2*(i+1)] = softmax(tau_or[:, 2*i:2*(i+1)])
import scipy
print('pearson correlation of tau on test set: ', scipy.stats.pearsonr(sub_cons_correct, torch.sum(tmp_or, 0).cpu().detach().numpy()))            

#rint('Confusion matrix:')
#print(conf_mat)
    
acc = 100.0*float(correct)/total
c_acc = 100.0*float(constraint_correct)/total
group_acc = 100.0*float(group_ok)/total
print("| Test Result\tAcc@1: %.2f%%" %(acc))
print("| Test Result\tCAcc: %.2f%%" %(c_acc))
print("| Test Result\tGroupAcc: %.2f%%" %(group_acc))
