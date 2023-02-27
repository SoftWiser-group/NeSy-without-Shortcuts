############### Pytorch HWF configuration file ###############
import math

start_epoch = 1
sgd_epochs = 0
adam_epochs = 600 
num_epochs = sgd_epochs + adam_epochs
optim_type = 'SGD+ADAM'
batch_size = 128
sgd_lr = 1e-3
adam_lr = 1e-3 # 0.001 
tau_lr = 0.1
use_cuda = True

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
    
    print(dir(cls))
    return type('handwritten-dataset', (cls,), {
        '__getitem__': __getitem__,
    })
