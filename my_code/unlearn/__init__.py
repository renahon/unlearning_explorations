from .GA import GA,GA_l1
from .NegGradPlus import NegGradPlus
from .RL import RL
from .FT import FT,FT_l1

from .retrain import retrain
from .impl import load_unlearn_checkpoint, save_unlearn_checkpoint
from .FT_prune import FT_prune
from .RL_pro import RL_proximal
from .sebastian import Sebastian
from .irene import irene
from .fanchuan import fanchuan

def raw(data_loaders, model, criterion, args, mask=None):
    pass


def get_unlearn_method(name):
    """method usage:

    function(data_loaders, model, criterion, args)"""
    if name == "raw":
        return raw
    elif name == "RL":
        return RL
    elif name == "GA":
        return GA
    elif name == "FT":
        return FT
    elif name == "FT_l1":
        return FT_l1
    elif name == "retrain":
        return retrain
    elif name == "FT_prune":
        return FT_prune
        return GA_l1
    elif name=="NegGradPlus":
        return NegGradPlus
    elif name =='Sebastian':
        return Sebastian
    elif name == 'irene':
        return irene
    elif name == "fanchuan":
        return fanchuan
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented!")
