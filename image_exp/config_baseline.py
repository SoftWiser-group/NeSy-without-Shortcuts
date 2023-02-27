############### Pytorch CIFAR configuration file ###############
import math

start_epoch = 1
sgd_epochs = 800
num_epochs = sgd_epochs
optim_type = 'sgd'
batch_size = 128
adam_lr = 0.1
tau_lr = 0.1
use_cuda = True

mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
}

# Only for cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes = ('beaver', 'dolphin', 'otter', 'seal', 'whale','fish', 
            'aquarium fish', 'flatfish', 'ray', 'shark', 'trout', 
            'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
            'bottles', 'bowls', 'cans', 'cups', 'plates',
            'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
            'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
            'bed', 'chair', 'couch', 'table', 'wardrobe', 
            'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach', 
            'bear', 'leopard', 'lion', 'tiger', 'wolf', 
            'bridge', 'castle', 'house', 'road', 'skyscraper', 
            'cloud', 'forest', 'mountain', 'plain', 'sea', 
            'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
            'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
            'crab', 'lobster', 'snail', 'spider', 'worm',
            'baby', 'boy', 'girl', 'man', 'woman', 
            'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle', 
            'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel', 
            'maple', 'oak', 'palm', 'pine', 'willow', 
            'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train', 
            'lawn-mower', 'rocket', 'streetcar', 'tank')

def lr_adapt(lr, epoch):
    optim_factor = 30
    lr = optim_factor * lr
    return lr


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s

def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })
