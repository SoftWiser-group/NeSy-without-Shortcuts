import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.linear = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.linear(out)
        return out

    def penultimate_forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return out  

    def intermediate_forward(self, x, layer_index):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        if layer_index == 1:
            out = F.relu(self.conv2(out))
            out = F.max_pool2d(out, 2)
        elif layer_index == 2:
            out = F.relu(self.conv2(out))
            out = F.max_pool2d(out, 2)            
            out = out.view(out.size(0), -1)
            out = F.relu(self.fc1(out))
        elif layer_index == 3:
            out = F.relu(self.conv2(out))
            out = F.max_pool2d(out, 2)            
            out = out.view(out.size(0), -1)
            out = F.relu(self.fc1(out))            
            out = F.relu(self.fc2(out))        
        return out

    # function to extact the multiple features
    def feature_list(self, x):
        out_list = []
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out_list.append(out)   
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out_list.append(out)        
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out_list.append(out)   
        out = F.relu(self.fc2(out))
        out_list.append(out)
        y = self.linear(out)
        return y, out_list


# def test():
#     net = LeNet(num_classes=10)
#     x = torch.randn(1,1,32,32)
#     y = net(Variable(x))
#     print(y.size())
#     print(net.intermediate_forward(Variable(x), 2).shape)

# test()
