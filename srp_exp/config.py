############### Pytorch SRP configuration file ###############
import math

start_epoch = 1
sgd_epochs = 0
adam_epochs = 300 
num_epochs = sgd_epochs + adam_epochs
optim_type = 'SGD+ADAM'
batch_size = 128
sgd_lr = 1e-5
adam_lr = 1e-5 # 0.001 
tau_lr = 0.1
use_cuda = True

n = 15; hidden=1000; 

def lr_adapt(lr, epoch):
    optim_factor = 30
    lr = optim_factor * lr
    return lr


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s

def swap(A, x, y):
    tmp = A[x, :].clone()
    A[x, :] = A[y, :] 
    A[y, :] = tmp
    tmp = A[:, x].clone()
    A[:, x] = A[:, y] 
    A[:, y] = tmp    
    return A

import torch
import torch.nn as nn
import torch.nn.functional as F
class GCN(nn.Module):
    def __init__(self, nclass, N, H):
        super(GCN, self).__init__()

        self.fc1 = nn.Linear(N * N, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, H)
        self.fc4 = nn.Linear(H, N)

    def forward(self, adj):
        y = adj.view(-1)
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc3(y))
        y = self.fc4(y)
        y = F.relu(y)
        return y
