from __future__ import print_function


from utils import *
from dataset import *
import argparse
from NN_AOG import NNAOG

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

parser = argparse.ArgumentParser(description='PyTorch CIFAR-100 Logic Training')
parser.add_argument('--seed', default=1, type=int, help='Random seed to use.')
parser.add_argument('--dataset', default='cifar100', type=str, help='Data set.')
parser.add_argument('--data_used', default=1.00, type=float, help='percentage of data used')
parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint')
parser.add_argument('--tol', type=float, default=1e-2, help='Tolerance for constraints')
args = parser.parse_args()
sys.path.append('../')
from config import *

np.random.seed(args.seed)
torch.manual_seed(args.seed)
train_set = MathExprDataset('train', numSamples=int(10000*args.data_used), randomSeed=777)
test_set = MathExprDataset('test')
num_cons = len(train_set)
print('train:', len(train_set), '  test:', len(test_set))
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                        shuffle=False, num_workers=2, collate_fn=MathExpr_collate)
eval_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                        shuffle=False, num_workers=2, collate_fn=MathExpr_collate)

# file_name = './checkpoint/' + args.resume_from + '.t7'
# net = NNAOG()
# net.sym_net.load_state_dict(torch.load(file_name))
checkpoint = torch.load('./checkpoint/' + args.resume_from + '.t7')
net = checkpoint['net']
tau_and = checkpoint['tau_and']
tau_or = checkpoint['tau_or']
net.cuda()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

def cons_sat(outputs):
    batch_size = outputs.shape[0]
    cons = []
    or_sat = []
    l = outputs.shape[1]
    for i in range(l-1):
        or0 = EQ(outputs[:, i, -4:].sum(dim=-1) + outputs[:, i+1, -4:].sum(dim=-1), 1)
        or1 = EQ(outputs[:, i, :-4].sum(dim=-1) + outputs[:, i+1, :-4].sum(dim=-1), 2)
        or_res = BatchOr([or1, or0], batch_size)
        cons.append(or_res)  
        or_sat.append(torch.stack((or1.satisfy(args.tol), or0.satisfy(args.tol)), 0))
    and_res = BatchAnd(cons, batch_size)
    ans_sat = and_res.satisfy(args.tol)
    if l == 1:
        ans_sat = torch.Tensor(torch.ones(batch_size,)) 
    return ans_sat, or_sat

def evaluate(model, dataloader):
    model.eval() 
    res_all = []
    res_pred_all = []
    
    expr_all = []
    expr_pred_all = []

    for sample in dataloader:
        img_seq = sample['img_seq']
        label_seq = sample['label_seq']
        res = sample['res']
        seq_len = sample['len']
        expr = sample['expr']
        img_seq = img_seq.to(device)
        label_seq = label_seq.to(device)

        masked_probs = model(img_seq)
        selected_probs, preds = torch.max(masked_probs, -1)
        selected_probs = torch.log(selected_probs+1e-12)
        expr_preds, res_preds = eval_expr(preds.data.cpu().numpy(), seq_len)

        res_pred_all.append(res_preds)
        res_all.append(res)
        expr_pred_all.extend(expr_preds)
        expr_all.extend(expr)
        

    res_pred_all = np.concatenate(res_pred_all, axis=0)
    res_all = np.concatenate(res_all, axis=0)
    print('Grammar Error: %.2f'%(np.isinf(res_pred_all).mean()*100))
    acc = equal_res(res_pred_all, res_all).mean()

    
    expr_pred_all = ''.join(expr_pred_all)
    expr_all = ''.join(expr_all)
    sym_acc = np.mean([x == y for x,y in zip(expr_pred_all, expr_all)])
    
    return acc, sym_acc

net.eval()
### training results
constraint_correct = 0
total = 0

np.set_printoptions(threshold=np.inf)
softmax = torch.nn.Softmax(dim=-1)
for batch_idx, sample in enumerate(train_dataloader):
    img_seq = sample['img_seq']
    label_seq = sample['label_seq']
    res = sample['res']
    seq_len = sample['len']
    expr = sample['expr']

    img_seq = img_seq.cuda()
    label_seq = label_seq.cuda()
    max_len = img_seq.shape[1]
    outputs = net(img_seq, islogits=True)
    probs = softmax(outputs)
    
    ans_sat, or_sat = cons_sat(probs)
    constraint_correct += ans_sat.sum()
    total += outputs.size(0)

acc, sym_acc = evaluate(net, train_dataloader)
    
c_acc = 100.0*float(constraint_correct)/total
print("| Train Result\tAcc@1: %.2f%%" %(100*float(acc)))
print("| Train Result\tCAcc: %.2f%%" %(c_acc))
print("| Train Result\tSymAcc: %.2f%%" %(100*float(sym_acc)))

### testing results
constraint_correct = 0
total = 0


np.set_printoptions(threshold=np.inf)
for batch_idx, sample in enumerate(eval_dataloader):
    img_seq = sample['img_seq']
    label_seq = sample['label_seq']
    res = sample['res']
    seq_len = sample['len']
    expr = sample['expr']

    img_seq = img_seq.cuda()
    label_seq = label_seq.cuda()
    max_len = img_seq.shape[1]
    outputs = net(img_seq, islogits=True)
    probs = softmax(outputs)
    
    ans_sat, or_sat = cons_sat(probs)
    constraint_correct += ans_sat.sum()
    total += outputs.size(0)

acc, sym_acc = evaluate(net, eval_dataloader)
    
c_acc = 100.0*float(constraint_correct)/total
print("| Test Result\tAcc@1: %.2f%%" %(100*float(acc)))
print("| Test Result\tCAcc: %.2f%%" %(c_acc))
print("| Test Result\tSymAcc: %.2f%%" %(100*float(sym_acc)))

