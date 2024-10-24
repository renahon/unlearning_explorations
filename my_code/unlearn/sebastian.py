import sys
import time
import torch.nn as nn
import torch
import utils
import torch.nn.utils.prune as prune
from .impl import iterative_unlearn
import copy
import torch.nn.functional as F
from math import sqrt


sys.path.append(".")
from imagenet import get_x_y_from_data_dict

def prune_model(net, amount=0.95, rand_init=True):
    # Modules to prune
    modules = list()
    for k, m in enumerate(net.modules()):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            modules.append((m, 'weight'))
            if m.bias is not None:
                modules.append((m, 'bias'))

    # Prune criteria
    prune.global_unstructured(
        modules,
        #pruning_method=prune.RandomUnstructured,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    # Perform the prune
    for k, m in enumerate(net.modules()):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            prune.remove(m, 'weight')
            if m.bias is not None:
                prune.remove(m, 'bias')

    # Random initialization
    if rand_init:
        for k, m in enumerate(net.modules()):
            if isinstance(m, nn.Conv2d):
                mask = m.weight == 0
                c_in = mask.shape[1]
                k = 1/(c_in*mask.shape[2]*mask.shape[3])
                randinit = (torch.rand_like(m.weight)-0.5)*2*sqrt(k)
                m.weight.data[mask] = randinit[mask]
            if isinstance(m, nn.Linear):
                mask = m.weight == 0
                c_in = mask.shape[1]
                k = 1/c_in
                randinit = (torch.rand_like(m.weight)-0.5)*2*sqrt(k)
                m.weight.data[mask] = randinit[mask]

def sebastian_loss(outputs,original_outputs):
    softmax_outputs = torch.softmax(outputs,dim=1)
    softmax_original_outputs = torch.softmax(original_outputs,dim=1)
    log_softmax_outputs = torch.log_softmax(outputs,dim=1)
    log_softmax_original_outputs = torch.log_softmax(original_outputs,dim=1)
    entropy = -torch.sum(softmax_outputs*log_softmax_outputs,dim=1)
    original_entropy = -torch.sum(softmax_original_outputs*log_softmax_original_outputs,dim=1)
    return F.mse_loss(entropy,original_entropy)

def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def Sebastian_iter(
    data_loaders, model, criterion, optimizer, epoch, args, mask=None, with_l1=False
):
    train_loader = data_loaders["retain"]
    val_loader = data_loaders["val"]
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()
    original_model = copy.deepcopy(model)
    start = time.time()
    if epoch==0:
        prune_amount = args.rate
        prune_model(model, prune_amount, True)
    else :
        for i, (image, target) in enumerate(train_loader):
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            image = image.cuda()
            target = target.cuda()
            if epoch < args.unlearn_epochs - args.no_l1_epochs:
                current_alpha = args.alpha * (
                    1 - epoch / (args.unlearn_epochs - args.no_l1_epochs)
                )
            else:
                current_alpha = 0
            # compute output
            output_clean = model(image)
            original_outputs = original_model(image)

            with torch.no_grad():
                original_outputs = original_model(image)
    
            loss = criterion(output_clean, target)
            seb_loss = sebastian_loss(output_clean,original_outputs)
            loss+=seb_loss

            if with_l1:
                loss = loss + current_alpha * l1_regularization(model)
            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()
    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg

























@iterative_unlearn
def Sebastian(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    return Sebastian_iter(data_loaders, model, criterion, optimizer, epoch, args, mask)
