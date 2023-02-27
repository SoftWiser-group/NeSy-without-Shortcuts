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

from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import sys
sys.path.append('../../')
import dl2lib as dl2
from logic_encoder import *
import random
import models

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
use_cuda = torch.cuda.is_available()
print(use_cuda)


from config import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser = dl2.add_default_parser_args(parser)
parser.add_argument('--seed', default=42, type=int, help='Random seed to use.')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--epochs', default=100, type=int, help='epochs')
parser.add_argument('--growing', default=0, type=int, help='epochs')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar100', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--exp_name', default='', type=str, help='experiment name')
parser.add_argument('--resume_from', type=str, default=None, help='resume from checkpoint')
parser.add_argument('--testOnly', action='store_true', help='Test mode with the saved model')
parser.add_argument('--constraint', type=str, choices=['DL2', 'Semantic', 'none'], default='none', help='constraint system to use')
parser.add_argument('--constraint-weight', '--constraint_weight', type=float, default=0.6, help='weight for constraint loss')
parser.add_argument('--num_labeled', default=1000, type=int, help='Number of labeled examples (per class!).')
parser.add_argument('--skip_labled', default=0, type=int, help='Number of labeled examples (per class!).')
parser.add_argument('--decrease-eps-weight', default=1.0, type=float, help='Number of labeled examples (per class!).')
parser.add_argument('--c-eps', default=0.05, type=float, help='Number of labeled examples (per class!).')
parser.add_argument('--increase-constraint-weight', default=1.0, type=float, help='Number of labeled examples (per class!).')
parser.add_argument('--tol', type=float, default=1e-2, help='Tolerance for constraints')
args = parser.parse_args()

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
criterion = nn.CrossEntropyLoss()
        
# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    softmax = torch.nn.Softmax(dim=1)  
    sigmoid = torch.nn.Sigmoid()  
    for batch_idx, (lab, ulab) in enumerate(zip(trainloader_lab, trainloader_unlab)):
        inputs_u, targets_u, index = ulab
        inputs_u, targets_u = Variable(inputs_u), Variable(targets_u)
        inputs_r = [rotateset[i][0] for i in index]
        inputs_r = torch.cat(inputs_r, dim=0).unsqueeze(1)
        inputs_r = Variable(inputs_r)
        index = underline[index]
        n_u = inputs_u.size()[0]
        if use_cuda:
            inputs_u = inputs_u.cuda() # GPU settings
            inputs_r = inputs_r.cuda()

        if lab is None:
            n = 0
            all_outputs = net(inputs_u, inputs_r)
        else:
            inputs, targets, _ = lab
            inputs, targets = Variable(inputs), Variable(targets)
            n = inputs.size()[0]
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
            all_outputs = net(torch.cat([inputs, inputs_u, inputs_r], dim=0))

        optimizer.zero_grad()
        outputs_u = all_outputs[n:n+n_u,]
        probs_u = softmax(outputs_u)
        outputs_r = all_outputs[n+n_u:,]
        probs_r = softmax(outputs_r)

        outputs = all_outputs[:n,]
        ce_loss = criterion(outputs, targets)  # Loss

        # updates tau
        if args.constraint == 'DL2':
            # optim_tau.zero_grad()
            or_res1 = dl2.Or([dl2.LEQ(probs_r[:,9]-probs_r[:,i], -eps) for i in [0,1,2,3,4,5,6,7,8]])
            and_res2 = dl2.And([dl2.GEQ(probs_u[:,6]-probs_u[:,i], eps) for i in [0,1,2,3,4,5,7,8,9]])
            dl2_one_group = dl2.Or([or_res1, and_res2])
            dl2_loss = dl2_one_group.loss(args).mean()
            constraint_loss = dl2_loss
            loss = ce_loss + (args.constraint_weight * args.increase_constraint_weight**epoch) * dl2_loss
        elif args.constraint == 'Semantic':
            # probs_r = sigmoid(outputs_r)
            # probs_u = sigmoid(outputs_u)
            probs_r = softmax(outputs_r)
            probs_u = softmax(outputs_u)
            # neg p
            loss1 = 0
            for i in range(9):
                # loss1 += probs_r[:,i] *  torch.prod(1-probs_r[:,:],dim=-1) / (1-probs_r[:,i])
                one_cast = torch.cat([torch.ones([n_u, i]), torch.zeros([n_u, 1]), torch.ones([n_u, 10-i-1])],dim=-1)
                if use_cuda:
                    one_cast = one_cast.cuda()
                loss1 += torch.prod((one_cast - probs_r).abs(), dim=-1)
            loss1 = - torch.log(loss1)
            # q
            loss2 = - torch.log(probs_u[:,6]) - torch.log(1-probs_u[:, :]).sum(dim=-1) + torch.log(1-probs_u[:,6])
            # p
            loss3 = - torch.log(probs_r[:,9]) - torch.log(1-probs_r[:, :]).sum(dim=-1) + torch.log(1-probs_r[:,9])
            
            # vee
            loss = torch.exp(-loss1) + torch.exp(-loss2-loss3)
            loss = - torch.log(loss)
            
            # wedge
            # loss = loss1 + loss2
            
            constraint_loss = loss.mean()
            loss = ce_loss + (args.constraint_weight) * constraint_loss
        else:
            constraint_loss = 0    
            loss = ce_loss

        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        # estimation
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
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
            'epoch': epoch,
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


print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))

elapsed_time = 0
for epoch in range(start_epoch, start_epoch+num_epochs):
    start_time = time.time()
    acc = train(epoch)
    if epoch % 400 == 0:
        save(acc, epoch, net)
    test(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(get_hms(elapsed_time)))

if best_model is not None:
    print('The overall best model is from epoch %02d-th' %(best_epoch))
    # save(best_acc, 'overall',  best_model, best_tau)
    
print('\n[Phase 4] : Testing the best model which derived from epoch %02d-th' %(best_epoch))
print('* Val results : Acc@1 = %.2f%%' %(best_acc))
