import sys
from json import dumps

import torch

from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.evaluate import get_confusion_matrix, get_fact_accuracy
from deepproblog.examples.MNIST.network import MNIST_Net
from deepproblog.heuristics import geometric_mean
from deepproblog.heuristics import NeuralHeuristic
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.utils import get_configuration, format_time_precise, config_to_string

import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from dpl_utils import *
sys.path.append('../')
from config import *
sys.path.append('../../')
from logic_encoder import *

state = torch.load("snapshot/net.pth")
network = state['net']

# net = model.networks["mnist_net"]
net = network.cuda()
net.eval()
eps = 0.1
tol = 1e-2
dataset = 'mnist'
softmax = torch.nn.Softmax(dim=1)  
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

    ans_sat1 = or_res1.satisfy(tol)
    ans_sat2 = and_res2.satisfy(tol)

    all_res = BatchOr([or_res1, and_res2], batch_size)
    ans_sat = all_res.satisfy(tol)
    return ans_sat1, ans_sat2, ans_sat

sys.stdout.write("| ")
if dataset == 'mnist':
    MNIST = dataset_with_indices(torchvision.datasets.MNIST)
    trainset = MNIST(root='../../data/mnist', train=True, download=True, transform=transform_train)
    rotateset = MNIST(root='../../data/mnist', train=False, download=True, transform=transform_rotate)
    testset = MNIST(root='../../data/mnist', train=False, download=False, transform=transform_test)
elif dataset == 'usps':
    USPS = dataset_with_indices(torchvision.datasets.USPS)
    trainset = USPS(root='../../data/usps', train=True, download=True, transform=transform_train)
    rotateset = USPS(root='../../data/usps', train=False, download=True, transform=transform_rotate)
    testset = USPS(root='../../data/usps', train=False, download=False, transform=transform_test)
num_classes = 10

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

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
        inputs_r = torch.cat(inputs_r, dim=0).unsqueeze(1).cuda()
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


dataset = 'usps'
sys.stdout.write("| ")
if dataset == 'mnist':
    MNIST = dataset_with_indices(torchvision.datasets.MNIST)
    trainset = MNIST(root='../../data/mnist', train=True, download=True, transform=transform_train)
    rotateset = MNIST(root='../../data/mnist', train=False, download=True, transform=transform_rotate)
    testset = MNIST(root='../../data/mnist', train=False, download=False, transform=transform_test)
elif dataset == 'usps':
    USPS = dataset_with_indices(torchvision.datasets.USPS)
    trainset = USPS(root='../../data/usps', train=True, download=True, transform=transform_train)
    rotateset = USPS(root='../../data/usps', train=False, download=True, transform=transform_rotate)
    testset = USPS(root='../../data/usps', train=False, download=False, transform=transform_test)
num_classes = 10

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

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
        inputs_r = torch.cat(inputs_r, dim=0).unsqueeze(1).cuda()
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