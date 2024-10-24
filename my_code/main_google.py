

"""Main file to launch unlearning evaluation using the competition metric."""

import copy
import os
import numpy as np
import torch
from torch import nn
from torch import optim
from unlearning_evaluation import metric
from unlearning_evaluation import train_lib
import utils


_DATA_DIR = flags.DEFINE_string(
    'data_dir',
    'unlearning/SURF',
    'Path to the dataset.',
)

_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir',
    './checkpoints',
    'Path to the checkpoint directory.',
)

_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    './outputs',
    'Path to the output directory.',
)

_NUM_MODELS = flags.DEFINE_integer(
    'num_models',
    1,
    'Number of models to train.',
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Info for training the original and retrained models:
NUM_EPOCHS = 30


def _get_confs(net, loader):
  """Returns the confidences of the data in loader extracted from net."""
  confs = []
  for inputs, targets in loader:
    inputs = inputs.to(DEVICE)
    logits = net(inputs)
    logits = logits.detach().cpu().numpy()
    _, conf = metric.compute_logit_scaled_confidence(logits, targets)
    confs.append(conf)
  confs = np.concatenate(confs, axis=0)
  return confs


def get_unlearned_and_retrained_confs_and_accs(
    data_loaders,
    class_weights,
    retrained_confs_path,
):
  
    train_loader, val_loader, test_loader, retain_loader, forget_loader = data_loaders
    """Returns the confidence and accuracies of unlearned and retrained models."""
    # Step 1) Get the confidences and accuracies under the unlearned models
    #######################################################################
    unlearned_confs_forget = []
    unlearned_retain_accs, unlearned_test_accs, unlearned_forget_accs = [], [], []

    # Reload the original model from which all unlearning runs will start.
    original_model = train_lib.train_or_reload_model(
        train_loader,
        val_loader,
        path=os.path.join(_CHECKPOINT_DIR.value, 'original_model.pth'),
        class_weights=class_weights,
        num_epochs=NUM_EPOCHS,
        do_saving=True,
    )

    for i in range(_NUM_MODELS.value):
    net = do_unlearning(
        retain_loader,
        forget_loader,
        val_loader,
        class_weights,
        original_model,
    )
    net.eval()

    # For this particular model, compute the forget set confidences.
    confs_forget = _get_confs(net, forget_loader_no_shuffle)
    unlearned_confs_forget.append(confs_forget)

    unlearned_confs_forget = np.stack(unlearned_confs_forget)

    # Step 2) Get the confidences and accuracies under the retrained models
    #######################################################################
    recompute = True
    retrained_confs_forget = []
    retrain_retain_accs, retrain_test_accs, retrain_forget_accs = [], [], []

    if os.path.exists(retrained_confs_path):
    loaded_results = np.load(retrained_confs_path)
    # retrained_confs is [num models, num examples].
    assert loaded_results['retrained_confs'].shape[0] == _NUM_MODELS.value
    retrained_confs_forget = loaded_results['retrained_confs']
    retrain_retain_accs = loaded_results['retrain_retain_accs']
    retrain_test_accs = loaded_results['retrain_test_accs']
    retrain_forget_accs = loaded_results['retrain_forget_accs']
    recompute = False

    if recompute:
    for i in range(_NUM_MODELS.value):
        path = os.path.join(_CHECKPOINT_DIR.value, str(i))
        net = train_lib.train_or_reload_model(
            retain_loader,
            val_loader,
            path,
            class_weights,
            num_epochs=NUM_EPOCHS,
            do_saving=i < _NUM_MODELS.value,
            min_save_epoch=20,
        )
        # For this particular model, compute the forget set confidences.
        confs_forget = _get_confs(net, forget_loader_no_shuffle)
        retrained_confs_forget.append(confs_forget)
        # For this particular model, compute the retain and test accuracies.

        retrain_retain_accs.append(accs['retain'])
        retrain_test_accs.append(accs['test'])
        retrain_forget_accs.append(accs['forget'])

    retrained_confs_forget = np.stack(retrained_confs_forget)

    np.savez(
        retrained_confs_path,
        retrained_confs=retrained_confs_forget,
        retrain_retain_accs=retrain_retain_accs,
        retrain_test_accs=retrain_test_accs,
        retrain_forget_accs=retrain_forget_accs,
    )

    return (
        unlearned_confs_forget,
        retrained_confs_forget,
        unlearned_retain_accs,
        unlearned_test_accs,
        unlearned_forget_accs,
        retrain_retain_accs,
        retrain_test_accs,
        retrain_forget_accs,
    )


def main(unused_args):

    if not os.path.isdir(_CHECKPOINT_DIR.value):
        os.mkdir(_CHECKPOINT_DIR.value)

    if not os.path.isdir(_OUTPUT_DIR.value):
        os.mkdir(_OUTPUT_DIR.value)
        retrained_confs_path = os.path.join(_OUTPUT_DIR.value, 'retrain_confs.npz')

    (
        train_loader,
        val_loader,
        test_loader,
        retain_loader,
        forget_loader,
        forget_loader_no_shuffle,
        class_weights,
    ) = surf.get_dataset(
        batch_size=64, quiet=False, dataset_path=_DATA_DIR.value
    )



    (
        unlearned_confs_forget,
        retrained_confs_forget,
        unlearned_retain_accs,
        unlearned_test_accs,
        unlearned_forget_accs,
        retrain_retain_accs,
        retrain_test_accs,
        retrain_forget_accs,
    ) = get_unlearned_and_retrained_confs_and_accs(
        train_loader,
        val_loader,
        test_loader,
        retain_loader,
        forget_loader,
        forget_loader_no_shuffle,
        class_weights,
        retrained_confs_path,
    )

    u_r_mean = np.mean(unlearned_retain_accs)
    u_t_mean = np.mean(unlearned_test_accs)
    u_f_mean = np.mean(unlearned_forget_accs)
    r_r_mean = np.mean(retrain_retain_accs)
    r_t_mean = np.mean(retrain_test_accs)
    r_f_mean = np.mean(retrain_forget_accs)


    forget_score = metric.compute_forget_score_from_confs(
        unlearned_confs_forget, retrained_confs_forget)

if __name__ == '__main__':
  main()
