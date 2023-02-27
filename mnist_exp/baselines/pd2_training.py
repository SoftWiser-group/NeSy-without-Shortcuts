from __future__ import print_function

import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
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

sys.path.append('../')
from config import *

import os
use_cuda = torch.cuda.is_available()

from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

parser = argparse.ArgumentParser(description='PyTorch CIFAR-100 Logic Training')
parser.add_argument('--seed', default=1, type=int, help='Random seed to use.')
parser.add_argument('--dataset', default='cifar100', type=str, help='Data set.')
parser.add_argument('--net_type', default='resnet50', type=str, help='Model')
parser.add_argument('--num_labeled', default=100, type=int, help='Number of labeled examples (per class!).')
parser.add_argument('--lr', type=float, default=5e-4, help='The learning rate of SGD')
parser.add_argument('--exp_name', default='', type=str, help='Experiment name')
parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint')
parser.add_argument('--trun', type=bool, default=False, help='Using truncated gaussian framework')
parser.add_argument('--z_sigma', type=float, default=1, help='The variance of gaussian')
parser.add_argument('--target_sigma', type=float, default=1e-1, help='The lower bound of variance')
parser.add_argument('--constraint', type=bool, default=False, help='Constraint system to use')
parser.add_argument('--constraint_weight', type=float, default=1.0, help='Constraint weight')
parser.add_argument('--tol', type=float, default=1e-3, help='Tolerance for constraints')
args = parser.parse_args()
opt = 'minmax'
args.epsilon=1e-4; args.weight_decay=1e-6; args.beta1=0.9; args.beta2=0.999
args.dd_optim='sgd'; args.ddlr=1e-2; args.dd_mom=0.9; args.dd_warmup_iter=1000
args.dd_constraint_wt=1; args.dd=True; args.dd_constant_lambda=0
args.dd_warmup_iter=0; args.dd_update_freq=10; args.dd_decay_after=5 # 432
args.dd_increase_freq_after=1; args.dd_increase_freq_by=1; args.dd_decay_lr=2

eps = 0.1

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
best_acc = 0
best_model = None
start_epoch, num_epochs, batch_size, optim_type = start_epoch, num_epochs, batch_size, optim_type
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

print("| Preparing MNIST dataset...")
sys.stdout.write("| ")
MNIST = dataset_with_indices(torchvision.datasets.MNIST)
trainset = MNIST(root='../../data/mnist', train=True, download=True, transform=transform_train)
rotateset = MNIST(root='../../data/mnist', train=True, download=True, transform=transform_rotate)
testset = MNIST(root='../../data/mnist', train=False, download=False, transform=transform_test)
num_classes = 10

num_train = len(trainset)

per_class = [[] for _ in range(10)]
for i in range(num_train):
    per_class[trainset[i][1]].append(i)

train_lab_idx = []
train_unlab_idx = []
valid_idx = []
    
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
for i in range(10):
    np.random.shuffle(per_class[i])
    split = int(np.floor(0.2 * len(per_class[i])))
    if i != 6:
        train_lab_idx += per_class[i][split:]
    elif i == 6:
        train_unlab_idx += per_class[i][split:]
    valid_idx += per_class[i][:split]

print('Total train[labeled]: ', len(train_lab_idx))
print('Total train[unlabeled]: ', len(train_unlab_idx))
print('Total valid: ', len(valid_idx))
num_cons = len(train_unlab_idx)
underline = np.arange(len(train_lab_idx)+len(train_unlab_idx)+len(valid_idx))
tmp = np.arange(num_cons)
underline[train_unlab_idx] = tmp

train_labeled_sampler = SubsetRandomSampler(train_lab_idx)
train_unlabeled_sampler = SubsetRandomSampler(train_unlab_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

unlab_batch = batch_size if args.constraint != 'none' else 1

trainloader_lab = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, sampler=train_labeled_sampler, num_workers=2)
trainloader_unlab = torch.utils.data.DataLoader(
    trainset, batch_size=unlab_batch, sampler=train_unlabeled_sampler, num_workers=2)
validloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, sampler=valid_sampler, num_workers=2)

def getNetwork(args):
    if args.net_type == 'lenet':
        net = models.LeNet(10)
        file_name = 'lenet'
    elif args.net_type == 'mlp':
        net = models.MLP(10)
        file_name = 'mlp'
    else:
        assert False
    file_name += '_' + str(args.seed) + '_' + args.exp_name
    return net, file_name


# cons operator
def initial_constriants():
    dd_param_list = []
    net.eval()

    for batch_idx, ulab in enumerate(trainloader_unlab):
        inputs_u, targets_u, index = ulab
        index = underline[index]
        inputs_u, targets_u = Variable(inputs_u), Variable(targets_u)
        n_u = inputs_u.size()[0]
        if use_cuda:
            inputs_u, targets_u = inputs_u.cuda(), targets_u.cuda() # GPU settings

        outputs = net(inputs_u)
        probs_u = softmax(outputs)        
        
        # constraint_loss = 0
        if args.constraint == True:
            for k in range(n_u):
                dd_param_list.append(0)
    dd_param_list = torch.Tensor(dd_param_list)
    return dd_param_list

def cons_loss(probs_u, probs_r, dd_param_list, index, n_u):
    constraint_loss = 0
    cons = []
    # or_res1 = BatchOr([LE(probs_u[:,6]-probs_u[:,i], -0.01) for i in [0,1,2,3,4,5,7,8,9]], index.shape[0], var_or[index,0:9])
    # l1 = [GE(probs_u[:,6]-probs_u[:,i], 0.01) for i in [0,1,2,3,4,5,7,8,9]]
    # l2 = [GE(probs_r[:,9]-probs_r[:,i], 0.01) for i in [0,1,2,3,4,5,6,7,8]]
    # and_res2 = BatchAnd(l1 + l2, batch_size, var_and[index,0:18])
    #
    or_res1 = BatchOr([LE(probs_r[:,9]-probs_r[:,i], -eps) for i in [0,1,2,3,4,5,6,7,8]], index.shape[0])
    and_res2 = BatchAnd([GE(probs_u[:,6]-probs_u[:,i], eps) for i in [0,1,2,3,4,5,7,8,9]], index.shape[0])
    # 
    # or_res1 = BatchAnd([LE(probs_r[:,9], 0.1), LE(probs_u[:,6], 0.1)], index.shape[0], var_and[index,0:2])
    # and_res2 = GE(probs_u[:,6],0.9)
    or_res = BatchOr([or_res1, and_res2], index.shape[0])
    hwx_loss = or_res.encode(opt)
    constraint_loss += (dd_param_list[index] * hwx_loss).sum()
    return constraint_loss

# Model
print('\n[Phase 2] : Model setup')
if args.resume_from is not None:
    # Load checkpoint
    print('| Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/' + args.resume_from + '.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']             
else:
    print('| Building net type [' + args.net_type + ']...')
    net, file_name = getNetwork(args)
    # net.apply(conv_init)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True   

dd_param_list = initial_constriants()
dd_param_list = dd_param_list.cuda()
print('\n| initial constraints')  

criterion = nn.CrossEntropyLoss()
        
# Training
def train(epoch, dd_param_list):
    train_loss = 0
    correct = 0
    total = 0
    num_iter = 0
    lambda_iter = 0
    last_lambda_update = 0

    softmax = torch.nn.Softmax(dim=1)  

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, optim_w.state_dict()['param_groups'][0]['lr'])) # lr 
    for batch_idx, (lab, ulab) in enumerate(zip(trainloader_lab, trainloader_unlab)):

        net.train()
        dd_param_list.require_grad = False
        inputs_u, targets_u, index = ulab
        index = underline[index]
        inputs_u, targets_u = Variable(inputs_u), Variable(targets_u)
        inputs_r = [rotateset[i][0] for i in index]
        inputs_r = torch.cat(inputs_r, dim=0).unsqueeze(1)
        inputs_r = Variable(inputs_r)
        n_u = inputs_u.size()[0]
        if use_cuda:
            inputs_u = inputs_u.cuda() # GPU settings
            inputs_r = inputs_r.cuda()

        if lab is None:
            n = 0
            all_outputs = net(inputs_u)
        else:
            inputs, targets, _ = lab
            inputs, targets = Variable(inputs), Variable(targets)
            n = inputs.size()[0]
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
            all_outputs = net(torch.cat([inputs, inputs_u, inputs_r], dim=0))

        outputs_u = all_outputs[n:n+n_u,]
        probs_u = softmax(outputs_u)
        outputs_r = all_outputs[n+n_u:,]
        probs_r = softmax(outputs_r)

        optim_w.zero_grad()
        # update w
        outputs = all_outputs[:n,]
        ce_loss = criterion(outputs, targets)  # Loss
        
        if args.constraint == True:
            constraint_loss = cons_loss(probs_u, probs_r,  dd_param_list, index, n_u)
            loss = ce_loss + args.dd_constraint_wt * constraint_loss
        else:
            loss = ce_loss
        loss.backward()  # Backward Propagation
        optim_w.step() # Optimizer update

        if (args.dd and (not args.dd_constant_lambda)):
            dd_optim.zero_grad()
            net.eval()
            dd_param_list.require_grad = True
            outputs_u = net(inputs_u)
            # logits_u = F.log_softmax(outputs_u)
            probs_u = softmax(outputs_u)                
            outputs_r = net(inputs_r)
            probs_r = softmax(outputs_r)
            constraint_loss = cons_loss(probs_u, probs_r, dd_param_list, index, n_u)
            closs = -1.0*args.dd_constraint_wt*constraint_loss
            closs.backward()
            dd_optim.step()

            # dd_param_list = torch.clamp(dd_param_list, min=0.0)

            lambda_iter += 1
            last_lambda_update = num_iter
            
            #increase dd_update_freq
            if (args.dd_increase_freq_after is not None) and (lambda_iter % args.dd_increase_freq_after == 0):
                args.dd_update_freq += args.dd_increase_freq_by 
            #
            lr_decay_after = 1.0
            if hasattr(args,'dd_decay_lr_after'):
                lr_decay_after = args.dd_decay_lr_after
                assert lr_decay_after >= 0
            #

            if args.dd_decay_lr == 1: 
                lr_decay_after = 1.0/lr_decay_after 
                for param_group in dd_optim.param_groups:
                    param_group['lr'] = param_group['lr']*math.sqrt(float(lr_decay_after*(lambda_iter-1)+1)/float(lr_decay_after*lambda_iter+1))
            elif args.dd_decay_lr == 2:
                lr_decay_after = 1.0/lr_decay_after 
                for param_group in dd_optim.param_groups:
                    param_group['lr'] = param_group['lr']*(float(lr_decay_after*(lambda_iter-1)+1)/float(lr_decay_after*lambda_iter+1))
            elif args.dd_decay_lr == 3:
                #exponential decay
                assert lr_decay_after <= 1
                for param_group in dd_optim.param_groups:
                    param_group['lr'] = param_group['lr']*lr_decay_after                            
            # 

        num_iter += 1


        # estimation
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().detach().sum()
        
        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tCE Loss: %.4f, Constraint Loss: %.4f Acc@1: %.3f%%'
                %(epoch, num_epochs, batch_idx+1,
                    (len(train_lab_idx)//batch_size)+1, ce_loss, constraint_loss, 100.*float(correct)/total))
        sys.stdout.flush()   

    return 100.*float(correct)/total

def save(acc, e, net, best=False):
    state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if best:
        e = int(400* math.ceil(( float(epoch) / 400)) )
        save_point = './checkpoint/' + file_name + '_' + str(e) + '_best' + '.t7'
        # save_point = './checkpoint/' + file_name + '_overall_' + '.t7'
    else:
        save_point = './checkpoint/' + file_name + '_' + str(e) + '_' + '.t7'
    torch.save(state, save_point)
    return net
    
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
    return ans_sat1, ans_sat2
        
def test(epoch):
    global best_acc, best_model, best_tau, best_epoch
    net.eval()
    test_loss = 0
    correct = 0
    constraint_correct1 = 0
    constraint_correct2 = 0
    constraint_num = 0
    total = 0
    for batch_idx, (inputs, targets, index) in enumerate(validloader):
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
            ans_sat1, ans_sat2 = cons_sat(probs_u, probs_r)
            constraint_correct1 += ans_sat1.sum()
            constraint_correct2 += ans_sat2.sum()
            constraint_num += len(ind)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Save checkpoint when best model
    acc = 100.*float(correct)/total
    cons_acc1 = 100.*float(constraint_correct1)/constraint_num
    cons_acc2 = 100.*float(constraint_correct2)/constraint_num
    total_acc = (acc) 
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%% Cons_Acc1/2: %.2f%%/%.2f%%" %(epoch, loss.item(), acc, cons_acc1, cons_acc2))

    if total_acc > best_acc:
        # print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
        best_model = save(acc, _, net, best=True)
        best_epoch = epoch
        best_acc = total_acc
    

elapsed_time = 0
print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Logical Constraint = ' + str(args.constraint))

optim_w = optim.Adam(net.parameters(), lr=args.lr, eps=args.epsilon,
                    weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))

dd_optim = None
dd_param_list = nn.Parameter(dd_param_list)
if args.dd_optim == 'adam':
    dd_optim = optim.Adam([dd_param_list], lr = args.ddlr, weight_decay = args.dd_weight_decay, betas = (args.beta1,args.beta2),eps=args.epsilon)
else:
    dd_optim = optim.SGD([dd_param_list], lr = args.ddlr, momentum = args.dd_mom)

for epoch in range(start_epoch, start_epoch+num_epochs):    
    start_time = time.time()

    acc = train(epoch, dd_param_list)
    test(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(get_hms(elapsed_time)))

if best_model is not None:
    print('The overall best model is from epoch %02d-th' %(best_epoch))
    # save(best_acc, 'overall',  best_model, best_tau)
    
print('\n[Phase 4] : Testing the best model which derived from epoch %02d-th' %(best_epoch))
print('* Val results : Acc@1 = %.2f%%' %(best_acc))
