from utils import *
from dataset import *
import argparse
from NN_AOG import NNAOG
import torch.nn.functional as F

import sys
sys.path.append('../')
from config import *

sys.path.append('../../')
from logic_encoder import *

# Dataloader
parser = argparse.ArgumentParser(description='PyTorch HWF Logic Training')
parser.add_argument('--seed', default=1, type=int, help='Random seed to use.')
parser.add_argument('--data_used', default=1.00, type=float, help='percentage of data used')
parser.add_argument('--num_unlabel', default=0.00, type=float, help='Number of unlabeled examples.')
parser.add_argument('--constraint', type=bool, default=False, help='Constraint system to use')
parser.add_argument('--constraint_weight', type=float, default=1.0, help='Constraint weight')
parser.add_argument('--tol', type=float, default=1e-2, help='Tolerance for constraints')
parser.add_argument('--exp_name', default='', type=str, help='Experiment name')
parser.add_argument('--trun', type=bool, default=True, help='Using truncated gaussian framework')
parser.add_argument('--z_sigma', type=float, default=10, help='The variance of gaussian')
parser.add_argument('--target_sigma', type=float, default=1e-1, help='The lower bound of variance')
args = parser.parse_args()

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
        # selected_probs = torch.log(selected_probs+1e-12)
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


# cons operator
def initial_constriants():
    var_or = [None for i in range(num_cons)] # not a good method, update later
    var_and = [None for i in range(num_cons)] # not a good method, update later
    net.eval()

    for batch_idx, sample in enumerate(unlab_dataloader):
        index = sample['index']
        index = underline[index]

        img_seq = sample['img_seq']
        img_seq = img_seq.cuda()
        outputs = net(img_seq)

        n_u = outputs.shape[0]
        # constraint_loss = 0
        if args.constraint == True:
            for k in range(n_u):
                cons = []
                for i in range(6):
                    or_res = Or([EQ(outputs[k, i, -4:].sum(dim=-1) + outputs[k, i+1, -4:].sum(dim=-1), 1), EQ(outputs[k, i, :-4].sum(dim=-1) + outputs[k, i+1, :-4].sum(dim=-1), 2)])
                    cons.append(or_res)
                    if var_or[index[k]] is None:
                        var_or[index[k]] = or_res.tau.numpy()
                    else:
                        var_or[index[k]] = np.append(var_or[index[k]], or_res.tau.numpy())
                and_res = And(cons)
                var_and[index[k]] = and_res.tau.numpy()
    return var_or, var_and

def cons_loss(outputs, index, n_u):
    constraint_loss = 0
    cons = []
    for i in range(6):
        or_res = BatchOr([EQ(outputs[:, i, -4:].sum(dim=-1) + outputs[:, i+1, -4:].sum(dim=-1), 1), \
                             EQ(outputs[:, i, :-4].sum(dim=-1) + outputs[:, i+1, :-4].sum(dim=-1), 2)], index.shape[0], var_or[index, 2*i:2*i+2])
        cons.append(or_res)  
    and_res = BatchAnd(cons, index.shape[0], var_and[index])
    hwx_loss = and_res.encode()
    if args.trun == True: 
        # maximum likelihood of truncated gaussians
        xi = (0 - hwx_loss) / args.z_sigma
        over = - 0.5 * xi.square() 
        tmp = torch.erf(xi / np.sqrt(2))
        under = torch.log(1 - tmp) 
        loss = -(over - under).mean()
        constraint_loss += loss     
    else:
        constraint_loss += hwx_loss.square().mean() / np.square(args.target_sigma)
    return constraint_loss, hwx_loss.detach().cpu().numpy()

def save(acc, e, net, tau, best=False):
    tau_and, tau_or = tau
    state = {
            'net': net,
            'acc': acc,
            'epoch': epoch,
            'tau_and': tau_and,
            'tau_or': tau_or,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if best:
        e = int(0)
        save_point = './checkpoint/' + file_name + '_' + str(e) + '_best' + '.t7'
        # save_point = './checkpoint/' + file_name + '_overall_' + '.t7'
    else:
        save_point = './checkpoint/' + file_name + '_' + str(e) + '_' + '.t7'
    torch.save(state, save_point)
    return net, tau

# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    list_hwx = []

    if args.constraint == True:
        global var_or, var_and, sigma
    softmax = torch.nn.Softmax(dim=-1)  

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, optim_w.state_dict()['param_groups'][0]['lr'])) # lr 
    for batch_idx, (sample, sample_u) in enumerate(zip(train_dataloader, unlab_dataloader)):
        index = sample_u['index']
        index = underline[index]
        img_seq = sample['img_seq']
        label_seq = sample['label_seq']
        n = img_seq.size()[0]
        img_seq_u = sample_u['img_seq']
        # label_seq_u = sample_u['label_seq']
        n_u = img_seq_u.size()[0]

        if use_cuda:
            img_seq_u = img_seq_u.cuda() # GPU settings

        if n == 0:
            all_outputs = net(img_seq_u)
        else:
            if use_cuda:
                img_seq, label_seq = img_seq.cuda(), label_seq.cuda() # GPU settings
            all_outputs = net(torch.cat([img_seq, img_seq_u], dim=0), islogits=True)

        outputs_u = all_outputs[n:,]
        # logits_u = F.log_softmax(outputs_u)
        probs_u = softmax(outputs_u)

        # updates tau
        if args.constraint == True:
            # optim_tau.zero_grad()
            temp_u = probs_u.clone().detach()
            constraint_loss, _ = cons_loss(temp_u, index, outputs_u.shape[0])
            constraint_loss.backward()
            with torch.no_grad():
                var_or = var_or - tau_lr * var_or.grad
                var_and = var_and + tau_lr * var_and.grad
                # var_or, var_and = nag.iter(var_and, var_or)
            var_or.requires_grad = True
            var_and.requires_grad = True
        else:
            constraint_loss = 0    

        optim_w.zero_grad()
        # update w
        outputs = all_outputs[:n,].reshape(-1, 14)
        targets = label_seq.reshape(-1)
        ce_loss = criterion(outputs, targets)
        
        if args.constraint == True:
            constraint_loss, hwx_loss = cons_loss(probs_u, index, outputs.shape[0])
            list_hwx.extend(hwx_loss)
            loss = ce_loss + args.constraint_weight * constraint_loss
        else:
            loss = ce_loss
        loss.backward()  # Backward Propagation
        optim_w.step() # Optimizer update

        # estimation
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tCE Loss: %.4f, Constraint Loss: %.4f Acc@1: %.3f%%'
                %(epoch, num_epochs, batch_idx+1,
                    len(train_dataloader), ce_loss, constraint_loss, 100.*float(correct)/total))
        sys.stdout.flush()   

    if scheduler is not None:
        scheduler.step()

    # # update sigma
    if args.constraint == True:
        # sigma = np.mean(np.square(list_hwx))
        # sigma = torch.tensor(np.sqrt(sigma))
        error = np.mean(np.array(list_hwx).reshape(-1))
        sigma = torch.tensor(np.square(error))
        sigma = torch.clamp(sigma, min=args.target_sigma, max=args.z_sigma)
        args.z_sigma = sigma
        print('\n Logic Error: %.3f, Update sigma: %.2f' %(error, sigma.detach().cpu().numpy()))

    return 100.*float(correct)/total

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

def test(epoch):
    global best_acc, best_model, best_tau, best_epoch
    net.eval()
    test_loss = 0
    correct = 0
    constraint_correct = 0
    total = 0
    softmax = torch.nn.Softmax(dim=-1)  
    for batch_idx, sample in enumerate(valid_dataloader):
        img_seq = sample['img_seq']
        label_seq = sample['label_seq']
        res = sample['res']
        seq_len = sample['len']
        expr = sample['expr']

        img_seq = img_seq.cuda()
        label_seq = label_seq.cuda()
        max_len = img_seq.shape[1]
        logits = net(img_seq, islogits=True)
        probs = softmax(logits)
        total += probs.size(0)
        ans_sat, _ = cons_sat(probs)
        constraint_correct += ans_sat.sum()

    # Save checkpoint when best model
    cons_acc = 100.*float(constraint_correct)/total
    # total_acc = acc
    acc, sym_acc = evaluate(net, valid_dataloader)
    acc = 100*float(acc)
    sym_acc = 100*float(sym_acc)
    print("\n| Validation Epoch #%d\t\t\t Acc@1: %.2f%% Cons_Acc: %.2f%% Sym_acc: %.2f%%" %(epoch, acc, cons_acc, sym_acc))

    total_acc = (sym_acc+cons_acc) / 2
    if total_acc > best_acc:
        print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(total_acc))
        best_model, best_tau = save(acc, _, net, [var_and, var_or], best=True)
        best_epoch = epoch
        best_acc = total_acc

np.random.seed(args.seed)
torch.manual_seed(args.seed)
train_set = MathExprDataset('train', numSamples=int(10000), randomSeed=777)
test_set = MathExprDataset('test')

from torch.utils.data.sampler import SubsetRandomSampler
train_lab_idx = []
train_unlab_idx = []
valid_idx = []
idx = np.arange(10000)
np.random.shuffle(idx)
split1 = int(np.floor(args.data_used * 10000))
split2 = int(np.floor((args.num_unlabel+args.data_used)*10000))
train_lab_idx = idx[0:split1]; train_unlab_idx = idx[split1:split2]; valid_idx = idx[split2:split2+1000]
num_cons = len(train_unlab_idx)
underline = np.arange(10000)
tmp = np.arange(num_cons)
underline[train_unlab_idx] = tmp

train_labeled_sampler = SubsetRandomSampler(train_lab_idx)
train_unlabeled_sampler = SubsetRandomSampler(train_unlab_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

print('train:', len(train_set), '  test:', len(test_set))
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, 
                        sampler=train_labeled_sampler,
                        num_workers=2, collate_fn=MathExpr_collate)
unlab_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, 
                        sampler=train_unlabeled_sampler,
                        num_workers=2, collate_fn=MathExpr_collate)
valid_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, 
                        sampler=valid_sampler,
                        num_workers=2, collate_fn=MathExpr_collate)
eval_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                        shuffle=False, num_workers=2, collate_fn=MathExpr_collate)
best_acc = 0
elapsed_time = 0
criterion = nn.CrossEntropyLoss()
file_name = 'sym_net' + '_' + str(args.seed) + '_' + args.exp_name
net = NNAOG()
# net.sym_net.load_state_dict(torch.load('../pretrain-sym_net/sym_net_acc50.ckpt'))
net.cuda()
if args.constraint == True:
    var_or, var_and = initial_constriants()
    if use_cuda:
        # device = torch.device('cuda')
        var_or = torch.tensor(var_or, requires_grad=True, device='cuda')
        var_and = torch.tensor(var_and, requires_grad=True, device='cuda')
        sigma = torch.tensor(args.z_sigma, device='cuda')
    else:
        var_or = torch.tensor(var_or, requires_grad=True)
        var_and = torch.tensor(var_and, requires_grad=True)
        sigma = torch.Tensor(args.z_sigma)
else:
    var_or = None
    var_and = None

# acc, sym_acc = evaluate(model, eval_dataloader)
for epoch in range(start_epoch, start_epoch+num_epochs):
    if epoch == start_epoch and sgd_epochs != 0:
        lr = sgd_lr
        optim_w = optim.SGD(net.parameters(), lr=lr)
        # scheduler = lr_scheduler.CosineAnnealingLR(optim_w, T_max=num_epochs, eta_min=1e-5)
        # scheduler = lr_scheduler.MultiStepLR(optim_w, milestones=[int(sgd_epochs/2), int(sgd_epochs * 3/4)], gamma=0.1)
        scheduler = None
    elif epoch == start_epoch + sgd_epochs:
        lr = adam_lr
        # optim_w = optim.Adam(net.parameters(), lr=lr)
        optim_w = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = None
        # update tau_lr
        tau_lr = lr_adapt(tau_lr, epoch)
    
    start_time = time.time()

    acc = train(epoch)
    if epoch % 100 == 0:
        if args.constraint == False:
            save(acc, epoch, net, [None, None])
        else:           
            save(acc, epoch, net, [var_and, var_or])           
    test(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(get_hms(elapsed_time)))