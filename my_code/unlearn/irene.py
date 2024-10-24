import utils
import time
from .impl import iterative_unlearn
from copy import deepcopy
import torch
import torch.nn.functional as F

def sebastian_loss(outputs,original_outputs):
    softmax_outputs = torch.softmax(outputs,dim=1)
    softmax_original_outputs = torch.softmax(original_outputs,dim=1)
    log_softmax_outputs = torch.log_softmax(outputs,dim=1)
    log_softmax_original_outputs = torch.log_softmax(original_outputs,dim=1)
    entropy = -torch.sum(softmax_outputs*log_softmax_outputs,dim=1)
    original_entropy = -torch.sum(softmax_original_outputs*log_softmax_original_outputs,dim=1)
    return F.mse_loss(entropy,original_entropy)



def irene_iter(data_loaders, model, PH,criterion, private_criterion,MI, optimizer, PH_optimizer, epoch, args,phase, original_model=None,mask=None, with_l1=False):
    train_loader = data_loaders["retain"]
    val_loader = data_loaders["val"]
    forget_loader = data_loaders["forget"]


    losses = utils.AverageMeter()
    private_losses = utils.AverageMeter()
    MI_tot = utils.AverageMeter()
    private_top1 = utils.AverageMeter()
    top1 = utils.AverageMeter()
    # switch to train mode
    model.train()
    if phase == 'forgetting':
        forget_dataset = deepcopy(forget_loader.dataset)
        training_dataset = deepcopy(train_loader.dataset)
        modified_forget_targets = [(target,0) for target in forget_dataset.labels]
        modified_training_targets = [(target,1) for target in training_dataset.labels]
        forget_dataset.labels = modified_forget_targets
        training_dataset.labels = modified_training_targets
        train_dataset = torch.utils.data.ConcatDataset([forget_dataset,training_dataset])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        start = time.time()
        for i, (image, (target,private_target)) in enumerate(train_loader):
            if epoch < args.warmup:
                    utils.warmup_lr(
                        epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                    )

            image = image.cuda()
            target = target.cuda()
            private_target = private_target.cuda()
            if epoch < args.unlearn_epochs - args.no_l1_epochs:
                current_alpha = args.alpha * (
                    1 - epoch / (args.unlearn_epochs - args.no_l1_epochs)
                )
            else:
                current_alpha = 0
            output= model(image)
            output_private = PH()
            retain_output= output[private_target==1]
            retain_target= target[private_target==1]
            loss_task = criterion(retain_output, retain_target)
            loss_private = private_criterion(output_private, private_target)
            MI_loss = MI(PH, private_target)
            losses.update(loss_task.item(), image.size(0))
            private_losses.update(loss_private.item(), image.size(0))
            MI_tot.update(MI_loss.item(), image.size(0))
            loss = (1-args.alpha) * loss_task + args.alpha * MI_loss
            loss.backward()
            
            PH_optimizer.zero_grad()
            loss_private.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            PH_optimizer.step()
            PH_optimizer.zero_grad()
            prec1 = utils.accuracy(output, target)[0]
            prec1_private = utils.accuracy(output_private, private_target)[0]
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))
            private_top1.update(prec1_private.item(), image.size(0))
            private_losses.update(loss_private.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Private Accuracy {private_top1.val:.3f} ({private_top1.avg:.3f})\t"
                    "Private Loss {private_loss.val:.4f} ({private_loss.avg:.4f})\t"
                    "MI {MI_tot.val:.6f} ({MI_tot.avg:.6f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(train_loader), end - start, loss=losses, top1=top1, private_top1=private_top1, private_loss=private_losses, MI_tot=MI_tot
                    )
                )
                start = time.time()
    else : 
        start = time.time()
        for i, (image, target) in enumerate(train_loader):
            if epoch < args.warmup:
                    utils.warmup_lr(
                        epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                    )

            image = image.cuda()
            target = target.cuda()
            output= model(image)
            loss=0
            if args.with_cross_entropy:
                loss = +criterion(output, target)
            if  args.with_seb_loss :
                original_outputs = original_model(image)
                seb_loss = sebastian_loss(output,original_outputs)
                loss +=seb_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            prec1 = utils.accuracy(output, target)[0]
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
    print("train_accuracy {top1.avg:.3f}".format(top1=top1),f"phase : {phase}")

    return top1.avg


@iterative_unlearn
def irene(data_loaders, model, criterion, optimizer, epoch, args, mask, PH, PH_optimizer, MI, private_criterion,phase,original_model=None):
    return irene_iter(data_loaders, model,PH, criterion, private_criterion, MI, optimizer, PH_optimizer, epoch, args, phase, original_model)
