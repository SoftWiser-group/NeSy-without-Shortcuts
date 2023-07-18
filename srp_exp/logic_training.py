from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

import os
import sys
sys.path.append('pygcn/pygcn')
from utils import load_data, accuracy
from layers import GraphConvolution
from graphs import Graph
sys.path.append('../../')

import sys
sys.path.append('../')
from config import *

sys.path.append('../../')
from logic_encoder import *

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--n_train', type=int, default=300, help='Number of train samples.')
parser.add_argument('--n_valid', type=int, default=150, help='Number of valid samples.')
parser.add_argument('--constraint', type=bool, default=False, help='Constraint system to use')
parser.add_argument('--trun', type=bool, default=True, help='Using truncated gaussian framework')
parser.add_argument('--exp_name', default='', type=str, help='Experiment name')
parser.add_argument('--z_sigma', type=float, default=1, help='The variance of gaussian')
parser.add_argument('--target_sigma', type=float, default=1e-1, help='The lower bound of variance')
parser.add_argument('--constraint_weight', type=float, default=1.0, help='Constraint weight')
parser.add_argument('--tol', type=float, default=1, help='Tolerance for constraints')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

# Load data

print('Generating train set...')
train_graphs, valid_graphs, test_graphs = [], [], []
for it in range(args.n_train):
    m = np.random.randint(n-1, int(n*(n-1)/2+1))
    train_graphs.append(Graph.gen_random_graph(n, m))

print('Generating valid set...')
for it in range(args.n_valid):
    m = np.random.randint(n-1, int(n*(n-1)/2+1))
    valid_graphs.append(Graph.gen_random_graph(n, m))

print('Generating test set...')
for it in range(args.n_valid):
    m = np.random.randint(n-1, int(n*(n-1)/2+1))
    test_graphs.append(Graph.gen_random_graph(n, m))

# Model and optimizer
net = GCN(nclass=1, N=n, H=hidden)
if use_cuda: net.cuda()

def initial_constriants():
    var_or = [None for i in range(num_cons)] # not a good method, update later
    var_and = [None for i in range(num_cons)] # not a good method, update later
    net.eval()

    for batch_idx, g in enumerate(train_graphs):
        with torch.no_grad():
            adj = torch.FloatTensor(g.input)
            if use_cuda:
                adj = adj.cuda()
        
        out = net(adj)
        n_u = 1
        # constraint_loss = 0
        if args.constraint == True:
            for j in range(n_u):
                cons = []
                for k in range(g.n):
                    adj_tmp = adj.clone()
                    adj_tmp = swap(adj_tmp, 0, k)
                    temp_out = net(adj_tmp).clone().detach()
                    for i in range(out.shape[0]):
                        if i == k:
                            eq_res = EQ(out[i], temp_out[i]) # snote i=k
                            cons.append(eq_res)
                        else:
                            leq_res = LE(out[i], out[k]+temp_out[i]) 
                            cons.append(leq_res)
                and_res = And(cons)
                if var_and[batch_idx] is None:
                    var_and[batch_idx] = and_res.tau.numpy()
                else:
                    var_and[batch_idx] = np.append(var_and[batch_idx], and_res.tau.numpy())
    return var_or, var_and

def cons_loss(outputs, index, n_u):
    out, temp_out, k = outputs
    constraint_loss = 0
    cons = []
    for i in range(out.shape[0]):
        if i == k:
            eq_res = EQ(out[i], temp_out[i]) # snote i=k
            cons.append(eq_res)
        else:
            leq_res = LE(out[i], out[k]+temp_out[i]) 
            cons.append(leq_res)
        # and_res = BatchAnd(cons, 1, var_and[index])
        and_res = And(cons, tau=var_and[index])
    hwx_loss = and_res.encode()
    if args.trun == True: 
        # maximum likelihood of truncated gaussians
        xi = (0 - hwx_loss) / args.z_sigma
        over = - 0.5 * xi.square() 
        tmp = torch.erf(xi / np.sqrt(2))
        under = torch.log(1 - tmp) 
        loss = -(over - under).mean()
        constraint_loss += loss     
    else:
        constraint_loss += hwx_loss.square().mean() / np.square(args.target_sigma)
    return constraint_loss, hwx_loss.detach().cpu().numpy()
        
def train(epoch):
    tot_err, tot_dl2_loss = 0, 0
    idx = np.arange(len(train_graphs))
    np.random.shuffle(idx)

    net.train()
    train_loss = 0
    correct = 0
    total = 0
    list_hwx = []

    if args.constraint == True:
        global var_or, var_and, sigma

    for i in range(idx.shape[0]):
        index = idx[i]
        g = train_graphs[index]

        with torch.no_grad():
            adj = torch.FloatTensor(g.input)
            if use_cuda:
                adj = adj.cuda()
        out = net(adj)
        # updates tau
        if args.constraint == True:
            constraint_loss = 0
            orig_out = out.clone().detach()
            for k in range(1, g.n):
                adj_tmp = adj.clone()
                adj_tmp = swap(adj_tmp, 0, k)
                temp_out = net(adj_tmp).clone().detach()
                tmp_loss, _ = cons_loss((orig_out, temp_out, k), index, out.shape[0])
                constraint_loss += tmp_loss
            constraint_loss.backward()
            with torch.no_grad():
                # var_or = var_or - tau_lr * var_or.grad
                var_and = var_and + tau_lr * var_and.grad
            # var_or.requires_grad = True
            var_and.requires_grad = True
        else:
            constraint_loss = 0    

        optim_w.zero_grad()
        dist = torch.FloatTensor([g.p[0, i] for i in range(g.n)])
        if use_cuda:
            dist = dist.cuda()

        err = torch.mean((dist - out) * (dist - out))
        tot_err += err.detach()
        
        if args.constraint == True:
            constraint_loss = 0
            hwx_loss = 0
            for k in range(1, g.n):
                adj_tmp = adj.clone()
                adj_tmp = swap(adj_tmp, 0, k)
                temp_out = net(adj_tmp)
                tmp_loss, tmp_hwx = cons_loss((out, temp_out, k), index, out.shape[0])
                constraint_loss += tmp_loss
                hwx_loss += tmp_hwx
            list_hwx.append(hwx_loss)
            loss = err + args.constraint_weight * constraint_loss
        else:
            loss = err
        loss.backward()  # Backward Propagation
        optim_w.step() # Optimizer update

        # estimation
        train_loss += loss.item()
        total += 1
        
        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\t MSE Loss: %.4f, Constraint Loss: %.4f'
                %(epoch, num_epochs, i+1,
                    (len(train_graphs)//1)+1, tot_err/total, constraint_loss))
        sys.stdout.flush()   

    if scheduler is not None:
        scheduler.step()

    # update sigma
    if args.constraint == True:
        # sigma = np.mean(np.square(list_hwx))
        # sigma = torch.tensor(np.sqrt(sigma))
        error = np.mean(list_hwx)
        sigma = torch.tensor(np.square(error))
        sigma = torch.clamp(sigma, min=args.target_sigma, max=args.z_sigma)
        args.z_sigma = sigma
        print('\n Logic Error: %.3f, Update sigma: %.2f' %(error, sigma.detach().cpu().numpy()))

    return 100.*float(correct)/total

def cons_sat(output, index):
    out, temp_out, k = output
    cons = []
    for i in range(out.shape[0]):
        if i == k:
            eq_res = EQ(out[i], temp_out[i]) # snote i=k
            cons.append(eq_res)
        else:
            leq_res = LE(out[i], out[k]+temp_out[i]) 
            cons.append(leq_res)
        # and_res = BatchAnd(cons, 1, var_and[index])
        and_res = And(cons)
    ans_sat = and_res.satisfy(args.tol)
    return ans_sat

def test(val=True, e=None):
    global best_err
    net.eval()
    tot_err = 0
    total = 0
    constraint_correct = 0
    ans_res = 0
    num_cons = 0
    tot_mae = 0

    for i, g in enumerate(valid_graphs if val else test_graphs):
        net.eval()
        with torch.no_grad():
            adj = torch.FloatTensor(g.input)

        if use_cuda:
            adj = adj.cuda()

        out = net(adj)
        dist = torch.FloatTensor([g.p[0, i] for i in range(g.n)])
        if use_cuda:
            dist = dist.cuda()

        err = torch.mean((dist - out) * (dist - out))
        mae = torch.mean((dist-out).abs())
        tot_err += err
        tot_mae += mae


        for k in range(1, g.n):
            adj_tmp = adj.clone()
            adj_tmp = swap(adj_tmp, 0, k)
            temp_out = net(adj_tmp)
            tmp_res = cons_sat((out, temp_out, k), i)
            ans_res += tmp_res
            num_cons += 1

        total += 1
        constraint_correct += ans_res

    print('[Valid] Average error: %.4f MAE: %.4f Cons_Acc: %.2f%%' % (tot_err/float(total), tot_mae/float(total), constraint_correct/float(num_cons)))

    error = tot_err / float(total)
    if error < best_err:
        best_err = error
        cons_acc = constraint_correct/float(num_cons)
        save(error, '0', net, [var_and, var_or], best=True)

def save(acc, e, net, tau, best=False):
    tau_and, tau_or = tau
    state = {
            'net': net,
            'acc': acc,
            'epoch': epoch,
            'tau_and': tau_and,
            'tau_or': tau_or,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if best:
        e = int(0)
        save_point = './checkpoint/' + file_name + '_' + str(e) + '_best' + '.t7'
        # save_point = './checkpoint/' + file_name + '_overall_' + '.t7'
    else:
        save_point = './checkpoint/' + file_name + '_' + str(e) + '_' + '.t7'
    torch.save(state, save_point)
    return net, tau

num_cons = len(train_graphs)

best_err = 1e5
cons_acc = 0
elapsed_time = 0
file_name = 'mlp_net' + '_' + str(args.seed) + '_' + args.exp_name
if args.constraint == True:
    var_or, var_and = initial_constriants()
    if use_cuda:
        # device = torch.device('cuda')
        var_and = torch.tensor(var_and, requires_grad=True, device='cuda')
        sigma = torch.tensor(args.z_sigma, device='cuda')
    else:
        var_and = torch.tensor(var_and, requires_grad=True)
        sigma = torch.Tensor(args.z_sigma)
else:
    var_or = None
    var_and = None

# acc, sym_acc = evaluate(model, eval_dataloader)
for epoch in range(start_epoch, start_epoch+num_epochs):
    if epoch == start_epoch and sgd_epochs != 0:
        lr = sgd_lr
        optim_w = optim.SGD(net.parameters(), lr=lr)
        # scheduler = lr_scheduler.CosineAnnealingLR(optim_w, T_max=num_epochs, eta_min=1e-5)
        # scheduler = lr_scheduler.MultiStepLR(optim_w, milestones=[int(sgd_epochs/2), int(sgd_epochs * 3/4)], gamma=0.1)
        scheduler = None
    elif epoch == start_epoch + sgd_epochs:
        lr = adam_lr
        optim_w = optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
        scheduler = None
        # update tau_lr
        tau_lr = lr_adapt(tau_lr, epoch)
    
    start_time = time.time()

    acc = train(epoch)
    if epoch % 100 == 0:
        if args.constraint == False:
            save(acc, epoch, net, [None, None])
        else:           
            save(acc, epoch, net, [var_and, var_or])           
    test(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(get_hms(elapsed_time)))

print('best error and cons are {}, {}'.format(best_err, cons_acc))

test(val=False)
