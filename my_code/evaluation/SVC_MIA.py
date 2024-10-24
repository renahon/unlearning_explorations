import numpy as np
import torch
import torch.nn.functional as F
from imagenet import get_x_y_from_data_dict
from sklearn.svm import SVC
from evaluation.metrics import false_positive_rate, false_negative_rate
from sklearn.metrics import make_scorer
from sklearn import model_selection

def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)


def m_entropy(p, labels, dim=-1, keepdim=False):
    log_prob = torch.where(p > 0, p.log(), torch.tensor(1e-30).to(p.device).log())
    reverse_prob = 1 - p
    log_reverse_prob = torch.where(
        reverse_prob > 0, reverse_prob.log(), torch.tensor(1e-30).to(p.device).log()
    )
    modified_probs = p.clone()
    modified_probs[:, labels] = reverse_prob[:, labels]
    modified_log_probs = log_reverse_prob.clone()
    modified_log_probs[:, labels] = log_prob[:, labels]
    return -torch.sum(modified_probs * modified_log_probs, dim=dim, keepdim=keepdim)


def collect_prob(data_loader, model):
    if data_loader is None:
        return torch.zeros([0, 10]), torch.zeros([0])

    prob = []
    targets = []

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            try:
                batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
                data, target = batch
            except:
                device = (
                    torch.device("cuda:0")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
                data, target = get_x_y_from_data_dict(batch, device)
            with torch.no_grad():
                output = model(data)
                prob.append(F.softmax(output, dim=-1).data)
                targets.append(target)

    return torch.cat(prob), torch.cat(targets)


def SVC_fit_predict(shadow_train, shadow_test, forget_target):
    n_shadow_train = shadow_train.shape[0]
    n_shadow_test = shadow_test.shape[0]
    n_forget_target = forget_target.shape[0]

    X_shadow = (
        torch.cat([shadow_train, shadow_test])
        .cpu()
        .numpy()
        .reshape(n_shadow_train + n_shadow_test, -1)
    )
    Y_shadow = np.concatenate([np.ones(n_shadow_train), np.zeros(n_shadow_test)])

    clf = SVC(C=3, gamma="auto", kernel="rbf")
    clf.fit(X_shadow, Y_shadow)

    X_forget = forget_target.cpu().numpy().reshape(n_forget_target, -1)
    acc = 1 - clf.predict(X_forget)

    return acc


def SVC_MIA(shadow_train, shadow_test, forget_loader, model, gold_model):
    shadow_train_prob, shadow_train_labels = collect_prob(shadow_train, model)
    shadow_test_prob, shadow_test_labels = collect_prob(shadow_test, model)
    forget_target_prob, forget_target_labels = collect_prob(forget_loader, model)


    shadow_gold_train_prob, shadow_gold_train_labels = collect_prob(shadow_train, gold_model)
    shadow_gold_test_prob, shadow_gold_test_labels = collect_prob(shadow_test, gold_model)
    forget_gold_target_prob, gold_target_labels = collect_prob(forget_loader, gold_model)



    shadow_train_conf = torch.gather(shadow_train_prob, 1, shadow_train_labels[:, None])
    shadow_test_conf = torch.gather(shadow_test_prob, 1, shadow_test_labels[:, None])
    forget_target_conf = torch.gather(forget_target_prob, 1, forget_target_labels[:, None])

    shadow_gold_train_conf = torch.gather(shadow_gold_train_prob, 1, shadow_gold_train_labels[:, None])
    shadow_gold_test_conf = torch.gather(shadow_gold_test_prob, 1, shadow_gold_test_labels[:, None])
    forget_gold_target_conf = torch.gather(forget_gold_target_prob, 1, gold_target_labels[:, None])

    shadow_train_entr = entropy(shadow_train_prob)
    shadow_test_entr = entropy(shadow_test_prob)
    forget_target_entr = entropy(forget_target_prob)

    shadow_gold_train_entr = entropy(shadow_gold_train_prob)
    shadow_gold_test_entr = entropy(shadow_gold_test_prob)
    forget_gold_target_entr = entropy(forget_gold_target_prob)

    acc_conf = SVC_fit_predict(
        shadow_train_conf, shadow_test_conf, forget_target_conf
    )
    acc_entr = SVC_fit_predict(
        shadow_train_entr, shadow_test_entr, forget_target_entr
    )

    acc_prob = SVC_fit_predict(
        shadow_train_prob, shadow_test_prob, forget_target_prob
    )



    gold_acc_conf = SVC_fit_predict(
        shadow_gold_train_conf, shadow_gold_test_conf, forget_gold_target_conf
    )
    gold_acc_entr = SVC_fit_predict(
        shadow_gold_train_entr, shadow_gold_test_entr, forget_gold_target_entr
    )
    gold_acc_prob = SVC_fit_predict(
        shadow_gold_train_prob, shadow_gold_test_prob, forget_gold_target_prob
    )

    mia_accuracy_conf = (gold_acc_conf == acc_conf).sum() / len(gold_acc_conf)
    fpr_conf = false_positive_rate(gold_acc_conf, acc_conf)
    fnr_conf = false_negative_rate(gold_acc_conf, acc_conf)

    mia_accuracy_entr = (gold_acc_entr == acc_entr).sum() / len(gold_acc_entr)
    fpr_entr = false_positive_rate(gold_acc_entr, acc_entr)
    fnr_entr = false_negative_rate(gold_acc_entr, acc_entr)

    mia_accuracy_prob = (gold_acc_prob == acc_prob).sum() / len(gold_acc_prob)
    fpr_prob = false_positive_rate(gold_acc_prob, acc_prob)
    fnr_prob = false_negative_rate(gold_acc_prob, acc_prob)

    gold_MIA_conf = gold_acc_conf.mean()
    MIA_conf = acc_conf.mean()

    gold_MIA_entr = gold_acc_entr.mean()
    MIA_entr = acc_entr.mean()

    gold_MIA_prob = gold_acc_prob.mean()
    MIA_prob = acc_prob.mean()

    results = {"conf" : {"gold_MIA": round(gold_MIA_conf,3), "MIA": round(MIA_conf,3),"MIA_accuracy" : round(mia_accuracy_conf,3)  ,"fpr": round(fpr_conf,3), "fnr": round(fnr_conf,3)},
                }




    
    return results
