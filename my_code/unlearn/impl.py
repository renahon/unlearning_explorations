import os
import time
import wandb
import matplotlib.pyplot as plt
import numpy as np
import pruner
import torch
import utils
from pruner import extract_mask, prune_model_custom, remove_prune
from imagenet import get_x_y_from_data_dict
from trainer import validate
from irene.utilities import *
from irene.core import *
import torch.nn as nn
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR


def plot_training_curve(training_result, save_dir, prefix):
    # plot training curve
    for name, result in training_result.items():
        plt.plot(result, label=f"{name}_acc")
    plt.legend()
    plt.savefig(os.path.join(save_dir, prefix + "_train.png"))
    plt.close()


def save_unlearn_checkpoint(model, evaluation_result, args):
    state = {"state_dict": model.state_dict(), "evaluation_result": evaluation_result}
    utils.save_checkpoint(state, False, args.save_dir, args.unlearn)
    utils.save_checkpoint(
        evaluation_result,
        False,
        args.save_dir,
        args.unlearn,
        filename="eval_result.pth.tar",
    )


def load_unlearn_checkpoint(model, device, args):
    checkpoint = utils.load_checkpoint(device, args.save_dir, args.unlearn)
    if checkpoint is None or checkpoint.get("state_dict") is None:
        return None

    current_mask = pruner.extract_mask(checkpoint["state_dict"])
    pruner.prune_model_custom(model, current_mask)
    pruner.check_sparsity(model)

    model.load_state_dict(checkpoint["state_dict"])

    # adding an extra forward process to enable the masks
    x_rand = torch.rand(1, 3, args.input_size, args.input_size).cuda()
    model.eval()
    with torch.no_grad():
        model(x_rand)

    evaluation_result = checkpoint.get("evaluation_result")
    return model, evaluation_result


def _iterative_unlearn_impl(unlearn_iter_func):
    def _wrapped(data_loaders, model, criterion, args, mask=None, **kwargs):
        decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
        device = torch.device("cuda:" + str(args.gpu))  
        if args.rewind_epoch != 0:
            initialization = torch.load(
                args.rewind_pth, map_location=device)
            current_mask = extract_mask(model.state_dict())
            remove_prune(model)
            # weight rewinding
            # rewind, initialization is a full model architecture without masks
            model.load_state_dict(initialization, strict=True)
            prune_model_custom(model, current_mask)
    
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.unlearn_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        if args.unlearn == "irene":
            original_model = copy.deepcopy(model)
            model.avgpool = nn.Sequential(model.avgpool, torch.nn.Identity().to(device))
            MI_loss = MI(device = device)
            PH_model = Privacy_head(model.avgpool, nn.Sequential(torch.nn.Linear(model.fc.in_features, 2))).to(device)        
            PH_optimizer = torch.optim.SGD(
                PH_model.parameters(),
                args.unlearn_lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay)
            private_criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight= torch.tensor([0.98,0.02])).to(device)

        if args.unlearn=='fanchuan':
            optimizer_retain = torch.optim.SGD(
                model.parameters(),
                args.unlearn_lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay)
            optimizer_forget = torch.optim.SGD(
                model.parameters(),
                lr =3e-4,
                momentum=args.momentum,
                weight_decay=args.weight_decay)
            total_step = int(len(data_loaders["forget"])*args.unlearn_epochs)
            fanchuan_scheduler = CosineAnnealingLR(optimizer_forget, T_max=total_step, eta_min=1e-6)


        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=decreasing_lr, gamma=args.gamma
        )  # 0.1 is fixed
        if args.rewind_epoch != 0:
            # learning rate rewinding
            for _ in range(args.rewind_epoch):
                scheduler.step()
        for epoch in range(0, args.unlearn_epochs):
            start_time = time.time()

            print(
                "Epoch #{}, Learning rate: {}".format(
                    epoch, optimizer.state_dict()["param_groups"][0]["lr"]
                )
            )
            if args.unlearn == "irene":
                if epoch>int(args.unlearn_epochs*2):
                    train_acc = unlearn_iter_func(
                    data_loaders, model, criterion, optimizer, epoch, args, mask,PH_model,PH_optimizer,MI_loss, private_criterion, 'training',original_model)
                else :
                    train_acc = unlearn_iter_func(
                    data_loaders, model, criterion, optimizer, epoch, args, mask,PH_model,PH_optimizer,MI_loss, private_criterion,'forgetting')    
            elif args.unlearn == "fanchuan":
                train_acc = unlearn_iter_func(
                    data_loaders, model, criterion, optimizer, optimizer_retain, optimizer_forget, fanchuan_scheduler, epoch, args)
            
            
            else:
                train_acc = unlearn_iter_func(
                    data_loaders, model, criterion, optimizer, epoch, args, mask
                )
            val_acc = validate(data_loaders["val"], model, criterion, args, type="val")
            forget_acc = validate(data_loaders["forget"], model, criterion, args, type = "forget")
            to_log = {
                "epoch": epoch,
                f"train_acc": train_acc,
                f"val_acc": val_acc,
                f"forget_acc": forget_acc,
            }        
            wandb.log(to_log)
            scheduler.step()

            print("one epoch duration:{}".format(time.time() - start_time))

    return _wrapped


def iterative_unlearn(func):
    """usage:

    @iterative_unlearn

    def func(data_loaders, model, criterion, optimizer, epoch, args)"""
    return _iterative_unlearn_impl(func)
