from abc import ABC
import numpy as np
import torch
from functools import reduce
# import argparse
# from .args import add_default_parser_args

# parser = argparse.ArgumentParser(description='logic encoder')
# parser = add_default_parser_args(parser)
# args = parser.parse_args()
# from .setting import args
# args = args()
# if args.precision == 'float32': 
#     DTYPE = torch.float32
# elif args.precision == 'float64':
#     DTYPE = torch.float64
DTYPE = torch.float64

softmax = torch.nn.Softmax(dim=1)

class Condition:

    def encode(self, **kwargs):
        return

    def satisfy(self, **kwargs):
        return

class GE(Condition):
    """ a >= b """

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.tol = 1e-3

    def encode(self):
        h = torch.clamp(self.b-self.a, min=0.0)
        Logic_loss = h
        return Logic_loss

    def satisfy(self, tol):
        return self.a - self.b > - tol

class LE(Condition):
    """ a <= b """

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def encode(self):
        h = torch.clamp(self.a-self.b, min=0.0)
        Logic_loss = h
        return Logic_loss

    def satisfy(self, tol):
        return self.b - self.a > - tol

class EQ(Condition):

    def __init__(self, a, b):
        self.a = a
        self.b = b
        
    def encode(self):
        h = torch.abs(self.a-self.b)
        Logic_loss = h
        return Logic_loss

    def satisfy(self, tol):
        return torch.abs(self.b - self.a) < tol

class NEQ(Condition):
    """ a != b """

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def encode(self):
        assert False, 'Encode NEQ is not supported!'
        # Logic_cons = torch.abs(self.a - self.b) < args.tol
        return 

    def satisfy(self, tol):
        return torch.abs(self.a - self.b) > tol

class And(Condition):
    """ E_1 & E_2 & ... E_k """
    """ version"""

    def __init__(self, exprs, tau=None):
        self.exprs = exprs
        if tau is None:
            self.tau = torch.tensor(np.ones([len(exprs), ]), dtype=DTYPE)
        else:
            self.tau = tau

    def grad(self):
        self.tau.requires_grad = True            

    def encode(self):
        # losses = [self.exprs[i].encode(args)*soft_tau[i] for i in range(len(self.exprs))]
        # Logic_loss = reduce(lambda x, y: x+y, losses)
        # losses = [exp.encode(args) for exp in self.exprs]
        # losses = torch.tensor(losses, requires_grad=True)
        # Logic_loss = torch.sum(losses * soft_tau)
        soft_tau = torch.nn.functional.softmax(self.tau, dim=0)
        losses = 0
        for i, exp in enumerate(self.exprs):
            losses += exp.encode() * soft_tau[i]
        Logic_loss = losses
        return Logic_loss

    def satisfy(self, tol):
        ret = None
        for exp in self.exprs:
            sat = exp.satisfy(tol)
            if not isinstance(sat, (np.ndarray, np.generic)):
                sat = sat.cpu().numpy()
            if ret is None:
                ret = sat.copy()
            ret = np.minimum(ret, sat) # False vs. True
        return ret

class BatchAnd(Condition):
    """ vec(x1) = 1 & vec(x2)=1 => vec(x11=1 & x21=1)"""
    """ version"""

    def __init__(self, exprs, batch_size, tau=None):
        self.exprs = exprs
        if tau is None:
            self.tau = torch.tensor(np.ones([batch_size, len(exprs)]), dtype=DTYPE)
        else:
            self.tau = tau

    def grad(self):
        self.tau.requires_grad = True            

    def encode(self):
        soft_tau = softmax(self.tau)
        losses = 0
        for i, exp in enumerate(self.exprs):
            losses += exp.encode() * soft_tau[:, i]
        Logic_loss = losses
        return Logic_loss

    def satisfy(self, tol):
        ret = None
        for exp in self.exprs:
            sat = exp.satisfy(tol)
            if not isinstance(sat, (np.ndarray, np.generic)):
                sat = sat.cpu().numpy()
            if ret is None:
                ret = sat.copy()
            ret = np.minimum(ret, sat) # False vs. True
        return ret
    
class Or(Condition):
    """ E_1 || E_2 || ... E_k """

    def __init__(self, exprs, tau=None):
        self.exprs = exprs
        if tau is None:
            self.tau = torch.tensor(np.ones([len(exprs), ]), dtype=DTYPE)
        else:
            self.tau = tau
    
    def grad(self):
        self.tau.requires_grad = True

    def encode(self):
        # losses = [exp.encode(args) for exp in self.exprs]
        # losses = torch.tensor(losses)
        # Logic_loss = - torch.sum(losses * soft_tau)
        soft_tau = torch.nn.functional.softmax(self.tau, dim=0)
        losses = 0
        for i, exp in enumerate(self.exprs):
            losses += exp.encode() * soft_tau[i]
        Logic_loss = losses
        return Logic_loss

    def satisfy(self, tol):
        ret = None
        for exp in self.exprs:
            sat = exp.satisfy(tol)
            if not isinstance(sat, (np.ndarray, np.generic)):
                sat = sat.cpu().numpy()
            if ret is None:
                ret = sat.copy()
            ret = np.maximum(ret, sat) # False vs. True
        return ret

class BatchOr(Condition):
    """ vec(x1) = 1 & vec(x2)=1 => vec(x11=1 & x21=1)"""
    """ version"""

    def __init__(self, exprs, batch_size, tau=None):
        self.exprs = exprs
        if tau is None:
            self.tau = torch.tensor(np.ones([batch_size, len(exprs)]), dtype=DTYPE)
        else:
            self.tau = tau

    def grad(self):
        self.tau.requires_grad = True            

    def encode(self):
        soft_tau = softmax(self.tau)
        losses = 0
        for i, exp in enumerate(self.exprs):
            losses += exp.encode() * soft_tau[:, i]
        Logic_loss = losses
        return Logic_loss

    def satisfy(self, tol):
        ret = None
        for exp in self.exprs:
            sat = exp.satisfy(tol)
            if not isinstance(sat, (np.ndarray, np.generic)):
                sat = sat.cpu().numpy()
            if ret is None:
                ret = sat.copy()
            ret = np.maximum(ret, sat) # False vs. True
        return ret        

class Implication(Condition):

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.t = Or([Negate(a), b])

    def loss(self):
        return self.t.encode()

    def satisfy(self, tol):
        return self.t.satisfy(tol)

class Negate(Condition):

    def __init__(self, exp):
        self.exp = exp

        # if isinstance(self.exp, LT):
        #     self.neg = GE(self.exp.a, self.exp.b)
        # elif isinstance(self.exp, GT):
        #     self.neg = LE(self.exp.a, self.exp.b)
        # elif isinstance(self.exp, EQ):
        #     self.cons = NEQ(self.exp.a, self.exp.b)
        # elif isinstance(self.exp, LEQ):
        #     self.neg, self.cons = GT(self.exp.a, self.exp.b)
        # elif isinstance(self.exp, GEQ):
        #     self.neg, self.cons = LT(self.exp.a, self.exp.b)
        if isinstance(self.exp, EQ):
            self.cons = NEQ(self.exp.a, self.exp.b)
        if isinstance(self.exp, NEQ):
            self.neg = EQ(self.exp.a, self.exp.b)
        elif isinstance(self.exp, LE):
            self.neg, self.cons = GE(self.exp.a, self.exp.b)
        elif isinstance(self.exp, GE):
            self.neg, self.cons = LE(self.exp.a, self.exp.b)        
        elif isinstance(self.exp, And):
            neg_exprs = [Negate(e) for e in self.exp.exprs]
            self.neg = Or(neg_exprs)
        elif isinstance(self.exp, Or):
            neg_exprs = [Negate(e) for e in self.exp.exprs]
            self.neg = And(neg_exprs)
        elif isinstance(self.exp, Implication):
            self.neg = And([self.exp.a, Negate(self.exp.b)])
        else:
            assert False, 'Class not supported %s' % str(type(exp))

    def encode(self):
        return self.neg.encode(), self.cons.encode() 

    def satisfy(self, tol):
        return np.maximum(self.neg.satisfy(tol), self.cons.satisfy(tol))







