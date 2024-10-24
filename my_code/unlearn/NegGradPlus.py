import sys
import time

import torch
import utils

from .impl import iterative_unlearn

sys.path.append(".")
from imagenet import get_x_y_from_data_dict


def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


@iterative_unlearn
def NegGradPlus(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    if args.imagenet_arch:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        for i, ((r_data),(f_data)) in enumerate(zip(retain_loader, forget_loader)):
            r_image, r_target = get_x_y_from_data_dict(r_data, device)
            f_image, f_target = get_x_y_from_data_dict(f_data, device)
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(retain_loader), args=args
                )

            # compute output
            r_output_clean = model(r_image)
            f_output_clean = model(f_image)
            r_loss = criterion(r_output_clean,r_target)
            f_loss = -criterion(f_output_clean,f_target)
            loss= args.alpha*r_loss + (1-args.alpha)*f_loss
            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()

            r_output = r_output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(r_output.data, r_target)[0]

            losses.update(loss.item(), r_image.size(0))
            top1.update(prec1.item(), r_image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(retain_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()
    else:
        for i, ((r_image, r_target),(f_image,f_target)) in enumerate(zip(retain_loader,forget_loader)):
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(retain_loader), args=args
                )

            r_image = r_image.cuda()
            r_target = r_target.cuda()
            f_image = f_image.cuda()
            f_target = f_target.cuda()
            

            # compute output
            r_output_clean = model(r_image)
            f_output_clean = model(f_image)
            r_loss = criterion(r_output_clean, r_target)
            f_loss = -criterion(f_output_clean, f_target)
            loss = args.alpha * r_loss + (1 - args.alpha) * f_loss

            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()

            r_output = r_output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(r_output.data, r_target)[0]

            losses.update(loss.item(), r_image.size(0))
            top1.update(prec1.item(), r_image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(retain_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg


@iterative_unlearn
def GA_l1(data_loaders, model, criterion, optimizer, epoch, args):
    train_loader = data_loaders["forget"]

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    for i, (image, target) in enumerate(train_loader):
        if epoch < args.warmup:
            utils.warmup_lr(
                epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
            )

        image = image.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(image)
        loss = -criterion(output_clean, target) + args.alpha * l1_regularization(model)

        optimizer.zero_grad()
        loss.backward()
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
