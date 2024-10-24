import torch
import torch.nn as nn
from copy import deepcopy
from .impl import iterative_unlearn
import utils
import time


def kl_loss_sym(x,y):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    return kl_loss(nn.LogSoftmax(dim=-1)(x),y)

@iterative_unlearn
def fanchuan(data_loaders, model, criterion, optimizer, optimizer_retain, optimizer_forget,fanchuan_scheduler, epoch, args) : 
    
    losses_forget = utils.AverageMeter()
    top1_forget = utils.AverageMeter()
    losses_retain = utils.AverageMeter()
    top1_retain = utils.AverageMeter()
    forget_loader = data_loaders['forget']
    retain_loader = data_loaders['retain']
    retain_ld = deepcopy(retain_loader)
    model.train()
    start = time.time()

    if epoch == 0 :
        for inputs,_ in forget_loader: ##First Stage 
            inputs = inputs.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            uniform_label = torch.ones_like(outputs).cuda() / outputs.shape[1] ##uniform pseudo label
            loss = kl_loss_sym(outputs, uniform_label) ##optimize the distance between logits and pseudo labels
            loss.backward()
            optimizer.step()
    else :
        for i, ((inputs_forget,labels_forget), (inputs_retain,labels_retain)) in enumerate(zip(forget_loader, retain_loader)):
            t = 1.15 
            inputs_forget, inputs_retain = inputs_forget.cuda(), inputs_retain.cuda()
            labels_forget, labels_retain = labels_forget.cuda(), labels_retain.cuda()
            optimizer_forget.zero_grad()
            outputs_forget,outputs_retain = model(inputs_forget),model(inputs_retain).detach()
            loss = (-1 * nn.LogSoftmax(dim=-1)(outputs_forget @ outputs_retain.T/t)).mean()
            loss.backward()
            optimizer_forget.step()
            fanchuan_scheduler.step()
            prec1 = utils.accuracy(outputs_forget, labels_forget)[0]
            top1_forget.update(prec1.item(), inputs_forget.size(0))
            losses_forget.update(loss.item(), inputs_forget.size(0))
            if (i + 1) % args.print_freq == 0:
                end = time.time()

                print(
                    "Phase : forgetting \t"
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(retain_loader), end - start, loss=losses_forget, top1=top1_forget
                    )
                )
                start = time.time()

        for i,(inputs,labels) in enumerate(retain_ld):
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer_retain.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_retain.step()
            prec1 = utils.accuracy(outputs, labels)[0]
            top1_retain.update(prec1.item(), inputs.size(0))
            losses_retain.update(loss.item(), inputs.size(0))
            if (i + 1) % args.print_freq == 0:
                end = time.time()

                print(
                    "Phase : Repair \t"
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(retain_loader), end - start, loss=losses_forget, top1=top1_retain
                    )
                )
                start = time.time()
        print("train_accuracy {top1.avg:.3f}".format(top1=top1_retain))


    return model