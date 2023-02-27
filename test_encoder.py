from logic_encoder import *
import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser(description='logic encoder testing')
parser = add_default_parser_args(parser)
args = parser.parse_args()


t = np.random.normal(size=[10,1])
t = torch.tensor(t)
softmax = torch.nn.Softmax(dim=0)
t = softmax(t)
# c = EQ(torch.sum(t),1)
# print(c.encode())
# c = GE(torch.sum(t[0:50]), 1)
# print(c.encode())
# c = GE(torch.sum(t)+2, 1)
# print(c.encode())
tau = torch.tensor(np.array([1.0, 1.0, 1.0, 1.0, 1.0]), requires_grad=True)
c1 = Or([LE(torch.sum(t[0:50]), 1), GE(torch.sum(t)+2, 1)], tau=tau[0:2])
c2 = Or([LE(torch.sum(t[50:100]), 1), GE(torch.sum(t)+2, 1)], tau=tau[3:5])
c = And([c1, c2])
# eval = c1.encode()+c2.encode()
eval = c.encode()
print(eval)
eval.backward()
print(tau.grad)


# print(c.encode()) 
# print(c.tau)
# c = Or([GE(torch.sum(t[0:50]), 1), GE(torch.sum(t)+2, 1)])
# eval = c.encode()
# vars = c.tau
# print(vars)
# eval.backward()
# print(vars.grad)

