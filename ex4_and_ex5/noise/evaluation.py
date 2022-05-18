import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import label_binarize


def evaluation(model, data_loader, categories=['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']):
    """ Generate evaluation report

    Returns:
        Summary on acc, recall, precision, f1-score, AUC
    """
    y, prob, label, pred = predict(model, data_loader)
    report = metrics.classification_report(label, pred, target_names=categories, digits=3)
    roc_auc = auc_scores(y, prob)
    report = "Evaluation Report:\n{}".format(report)
    report += "\nAUC: {}".format(roc_auc)
    return report


@torch.no_grad()
def predict(model, data_loader):
    """ Get predicted probabilities

    Returns:
        y: one-hot labels
        prob: predicted probabilities
    """
    model.eval()
    device = next(model.parameters()).device
    label = []
    prob = [[1 for _ in range(7)]]
    soft = nn.Softmax(dim=-1)

    for x, y, _ in data_loader:
        x, y = x.to(device), y.to(device)
        z = model(x)
        p = soft(z)
        prob = np.concatenate((prob, p.to('cpu')), axis=-2)
        label = np.concatenate((label, y.to('cpu')), axis=-1)
    
    prob = prob[1::]
    class_num = prob.shape[1]
    y = label_binarize(label, classes=[i for i in range(class_num)])
    pred = np.argmax(prob, axis=1)

    return y, prob, label, pred



def auc_scores(y, prob):
    class_num = prob.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Compute ROC curve and ROC area for each class
    for i in range(class_num):
        fpr[i], tpr[i], _ = metrics.roc_curve(y[:, i], prob[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area (computed globally)
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y.ravel(), prob.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area (simply average on each label)
    # aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(class_num)]))
    # interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(class_num):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # average it and compute AUC
    mean_tpr /= class_num

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    return roc_auc