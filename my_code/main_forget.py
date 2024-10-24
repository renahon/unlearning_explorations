import copy
import os
from collections import OrderedDict
import numpy as np
import arg_parser
import evaluation
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import unlearn
import utils
from trainer import validate
from evaluation import ToW
import wandb



def main():
    torch.cuda.empty_cache()
    args = arg_parser.parse_args()
    wandb_name = arg_parser.get_wandb_name(args)
    wandb_project = f"unlearnings_with_0.1_of_{args.dataset}"

    wandb.init(config=args,
        name=wandb_name,
        project=wandb_project)
    
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed
    if args.dataset == "femnist":
        (original_model,
        train_loader_full,
        val_loader,
        test_loader,
        marked_loader, svm_dataset_test) = utils.setup_model_dataset(args)
    else :
        (
        original_model,
        train_loader_full,
        val_loader,
        test_loader,
        marked_loader,
        ) = utils.setup_model_dataset(args)
    if args.unlearn!="retrain":          
        gold_model = copy.deepcopy(original_model)
        gold_model.cuda()
    original_model.cuda()
    def replace_loader_dataset(
        dataset, batch_size=args.batch_size, seed=1, shuffle=True
    ):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=shuffle,
        )

    forget_dataset = copy.deepcopy(marked_loader.dataset)

    try:
        marked = forget_dataset.labels < 0
        forget_dataset.data = forget_dataset.data[marked]
        forget_dataset.labels = -forget_dataset.labels[marked] - 1
        forget_loader = replace_loader_dataset(
            forget_dataset, seed=seed, shuffle=True
        )
        retain_dataset = copy.deepcopy(marked_loader.dataset)
        marked = retain_dataset.labels >= 0
        retain_dataset.data = retain_dataset.data[marked]
        retain_dataset.labels = retain_dataset.labels[marked]
        retain_loader = replace_loader_dataset(
            retain_dataset, seed=seed, shuffle=True
        )
        assert len(forget_dataset) + len(retain_dataset) == len(
            train_loader_full.dataset
        )
    except:
        marked = forget_dataset.targets < 0
        forget_dataset.imgs = forget_dataset.imgs[marked]
        forget_dataset.targets = -forget_dataset.targets[marked] - 1
        forget_loader = replace_loader_dataset(
            forget_dataset, seed=seed, shuffle=True
        )
        retain_dataset = copy.deepcopy(marked_loader.dataset)
        marked = retain_dataset.targets >= 0
        retain_dataset.imgs = retain_dataset.imgs[marked]
        retain_dataset.targets = retain_dataset.targets[marked]
        retain_loader = replace_loader_dataset(
            retain_dataset, seed=seed, shuffle=True
        )
        assert len(forget_dataset) + len(retain_dataset) == len(
            train_loader_full.dataset
        )

    print(f"number of retain dataset {len(retain_dataset)}")
    print(f"number of forget dataset {len(forget_dataset)}")
    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )

    criterion = nn.CrossEntropyLoss()
    evaluation_result = None

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        original_model, evaluation_result = checkpoint
        model = copy.deepcopy(original_model)
    else:
        checkpoint = torch.load(args.model_path, map_location=device)
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]

        if args.unlearn != "retrain":
            original_model.load_state_dict(checkpoint, strict=False)
            model = copy.deepcopy(original_model)
            gold_checkpoint = torch.load(args.gold_model_path, map_location=device)
            if "state_dict" in gold_checkpoint.keys():
                gold_checkpoint = gold_checkpoint["state_dict"]
            gold_model.load_state_dict(gold_checkpoint, strict=False)
        else :
            model = copy.deepcopy(original_model)
            original_model.load_state_dict(checkpoint, strict=False)


        mask = None
        if args.mask_path:
            mask = torch.load(args.mask_path, map_location=device)

        unlearn_method = unlearn.get_unlearn_method(args.unlearn)
        unlearn_method(unlearn_data_loaders, model, criterion, args, mask)
        unlearn.save_unlearn_checkpoint(model, None, args)

    if evaluation_result is None:
        evaluation_result = {}

    accuracy = {}
    for name, loader in unlearn_data_loaders.items():
        utils.dataset_convert_to_test(loader.dataset, args)
        val_acc = validate(loader, model, criterion, args, type = f"{name}_acc")
        original_val_acc = validate(loader, original_model, criterion, args, type = f"{name}_original_acc")
        accuracy[name+"_accuracy"] = val_acc
        accuracy[name+"_original_accuracy"] = original_val_acc
        if args.unlearn != "retrain":
            gold_val_acc = validate(loader, gold_model, criterion, args, type = f"{name}_gold_acc")
            accuracy[name + "_gold_accuracy"] = gold_val_acc
    evaluation_result["accuracy"] = accuracy
    unlearn.save_unlearn_checkpoint(model, evaluation_result, args)


    if "SVC_MIA_forget_efficacy" not in evaluation_result:
        if args.dataset=="femnist":
            shadow_test_loader = torch.utils.data.DataLoader(
                svm_dataset_test, batch_size=args.batch_size, shuffle=False
            )

            test_len = len(shadow_test_loader.dataset)
            utils.dataset_convert_to_test(retain_dataset, args)
            utils.dataset_convert_to_test(forget_loader, args)
            utils.dataset_convert_to_test(test_loader, args)

        else:    
            test_len = len(test_loader.dataset)
            forget_len = len(forget_dataset)
            retain_len = len(retain_dataset)

            utils.dataset_convert_to_test(retain_dataset, args)
            utils.dataset_convert_to_test(forget_loader, args)
            utils.dataset_convert_to_test(test_loader, args)
            shadow_test_loader = test_loader

        shadow_train = torch.utils.data.Subset(retain_dataset, list(range(test_len)))
        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=args.batch_size, shuffle=False
        )

        forget_loader = torch.utils.data.DataLoader(
            forget_loader.dataset, batch_size=args.batch_size, shuffle=False
        )
        if args.unlearn != "retrain":
                MIA = evaluation.SVC_MIA(
                    shadow_train_loader, shadow_test_loader, forget_loader, model, gold_model
                )
                original_MIA = evaluation.SVC_MIA(
                    shadow_train_loader, shadow_test_loader, forget_loader, original_model, gold_model
                )
                evaluation_result["SVC_MIA_forget_efficacy"] = MIA  
                evaluation_result["SVC_MIA_original_forget_efficacy"] = original_MIA
                print (f"forget efficacy: {evaluation_result['SVC_MIA_forget_efficacy']}")
                print (f"original forget efficacy: {evaluation_result['SVC_MIA_original_forget_efficacy']}")


    if args.unlearn != "retrain":
        Tow = ToW(
            evaluation_result["accuracy"]["forget_accuracy"],
            evaluation_result["accuracy"]["forget_gold_accuracy"],
            evaluation_result["accuracy"]["test_accuracy"],
            evaluation_result["accuracy"]["test_gold_accuracy"],
            evaluation_result["accuracy"]["retain_accuracy"],
            evaluation_result["accuracy"]["retain_gold_accuracy"],
        )
        original_Tow = ToW(
            evaluation_result["accuracy"]["forget_original_accuracy"],
            evaluation_result["accuracy"]["forget_gold_accuracy"],
            evaluation_result["accuracy"]["test_original_accuracy"],
            evaluation_result["accuracy"]["test_gold_accuracy"],
            evaluation_result["accuracy"]["retain_original_accuracy"],
            evaluation_result["accuracy"]["retain_gold_accuracy"],
        )
        print(f'ToW: {Tow}')
        print(f"original ToW: {original_Tow}")
        evaluation_result["ToW"] = Tow
        evaluation_result["original_ToW"] = original_Tow
    wandb.log(evaluation_result)


    unlearn.save_unlearn_checkpoint(model, evaluation_result, args)


if __name__ == "__main__":
    main()