import torch
import utils
from imagenet import get_x_y_from_data_dict


def validate(val_loader, model, criterion, args, type="val"):
    """
    Run evaluation
    """
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (image, target) in enumerate(val_loader):
        image = image.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = utils.accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

    print(      "{type} : \t"
                "Loss {loss.avg:.4f} \t"
                "Accuracy {top1.avg:.3f}".format(type=type, loss=losses, top1=top1)
            )

    return top1.avg


