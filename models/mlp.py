import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        self.fc1   = nn.Linear(32*32*1, 120)
        # self.fc2   = nn.Linear(120, 84)
        # self.linear = nn.Linear(84, num_classes)
        self.linear = nn.Linear(120,num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc1(out))
        # out = F.relu(self.fc2(out))
        out = self.linear(out)
        return out

