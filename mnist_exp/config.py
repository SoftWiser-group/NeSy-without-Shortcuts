############### Pytorch CIFAR configuration file ###############
import math

start_epoch = 1
sgd_epochs = 0
adam_epochs = 60
num_epochs = sgd_epochs + adam_epochs
optim_type = 'SGD+ADAM'
batch_size = 128
# sgd_lr = 0.1
# adam_lr = 0.001  # plz set 0.001 in resnet
tau_lr = 0.1
use_cuda = True

def lr_adapt(lr, epoch):
    optim_factor = 3000
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
