import itertools
import json
import random
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset as TorchDataset
from typing import Callable, List, Iterable, Tuple


from deepproblog.dataset import Dataset
from deepproblog.query import Query
from problog.logic import Term, list2term, Constant

_DATA_ROOT = Path(__file__).parent

transform_train = transforms.Compose([
        transforms.Resize([28,28]),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)), 
    ])

transform_rotate = transforms.Compose([
        transforms.Resize([28,28]),
        transforms.RandomRotation(degrees=[180,180]),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)), 
    ])

transform_test = transforms.Compose([
        transforms.Resize([28,28]),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)), 
    ])


datasets = {
    "train": torchvision.datasets.MNIST(
        root=str(_DATA_ROOT), train=True, download=True, transform=transform_train
    ),
    "rot": torchvision.datasets.MNIST(
        root=str(_DATA_ROOT), train=True, download=True, transform=transform_rotate
    ),
    "test": torchvision.datasets.MNIST(
        root=str(_DATA_ROOT), train=False, download=True, transform=transform_test
    ),
}




class MNIST_Images(object):
    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, item):
        return datasets[self.subset][int(item[0])][0]


MNIST_train = MNIST_Images("train")
MNIST_rotate = MNIST_Images("rot")
MNIST_test = MNIST_Images("test")


class MNIST(Dataset):
    def __len__(self):
        return len(self.data)

    def to_query(self, i):
        l = Constant(self.data[i][1])
        # DELETE LABEL 6
        if l != 6:
            return Query(
                Term("combine", Term("tensor", Term(self.dataset, Term("a"))),  Term("tensor", Term(self.rdataset, Term("a"))), l),
                substitution={Term("a"): Constant(i)},
            )
        else:
            return Query(
                Term("rotate", Term("tensor", Term(self.dataset, Term("a"))),  Term("tensor", Term(self.rdataset, Term("a")))),
                substitution={Term("a"): Constant(i)},
            )

    def __init__(self, dataset, rdataset=None):
        self.dataset = dataset
        self.data = datasets[dataset]
        if rdataset is not None:
            self.rdataset = rdataset
            self.rdata = datasets[rdataset]
        else:
            self.rdataset = dataset
            self.rdata = datasets[dataset]
