import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
import cv2

__all__ = ["Evaluation", "predict", "get_predictions", "probable", "get_probabilities",
           "get_report", "accuracy", "balanced_accuracy", "precision", "recall",
           "f1_score", "confusion_matrix", "roc_auc", "roc_curves"]

_device = torch.device("cpu")


@torch.no_grad()
def predict(model, x):
    model.eval()

    out = model(x)
    pred = torch.argmax(out, dim=-1)
    return pred


@torch.no_grad()
def get_predictions(model, data_loader, device=_device):
    model = model.to(device)

    preds = []
    targets = []
    for x, y in data_loader:
        preds.append(predict(model, x.to(device)))
        targets.append(y.to(device))

    return torch.cat(preds, dim=-1), torch.cat(targets, dim=-1)


@torch.no_grad()
def probable(model, x):
    model.eval()

    out = model(x)
    prob = F.softmax(out, dim=-1)
    return prob


@torch.no_grad()
def get_probabilities(model, data_loader, device=_device):
    model = model.to(device)

    probs = []
    targets = []
    for x, y in data_loader:
        probs.append(probable(model, x.to(device)))
        targets.append(F.one_hot(y.to(device), num_classes=x.shape[-1]))

    return torch.cat(probs, dim=-2), torch.cat(targets, dim=-2)


def get_report(preds, targets, categories=None):
    report = metrics.classification_report(targets, preds,
                                           target_names=categories,
                                           digits=3)
    return report


def accuracy(preds, targets):
    return metrics.accuracy_score(targets, preds)


def balanced_accuracy(preds, targets):
    return metrics.balanced_accuracy_score(targets, preds)


def precision(preds, targets, average=None):
    return metrics.precision_score(targets, preds, average=average)


def recall(preds, targets, average=None):
    return metrics.recall_score(targets, preds, average=average)


def f1_score(preds, targets, average=None):
    return metrics.f1_score(targets, preds, average=average)


def confusion_matrix(preds, targets):
    return metrics.confusion_matrix(targets, preds)


def roc_auc(probs, targets, average=None):
    return metrics.roc_auc_score(targets, probs, average=average)


def roc_curves(probs, targets):
    n_classes = probs.shape[-1]
    fpr = dict()
    tpr = dict()
    auc = dict()
    # 计算每个类别的ROC曲线和AUC面积
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(targets[:, i], probs[:, i])
        auc[i] = metrics.auc(fpr[i], tpr[i])

    # 计算ROC曲线和AUC面积的微观平均（micro-averaging）
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(targets.ravel(), probs.ravel())
    auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    # 计算宏观平均ROC曲线和AUC面积
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, auc


metric_dict = {
    "accuracy": accuracy,
    "acc": accuracy,
    "balanced_accuracy": balanced_accuracy,
    "b_acc": balanced_accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1_score,
    "confusion_matrix": confusion_matrix,
    "c_matrix": confusion_matrix,
    "roc_auc": roc_auc,
    "auc": roc_auc,
    "roc_curves": roc_curves
}

_pred_based = ["acc", "accuracy", "balanced_accuracy", "b_acc", "precision",
               "recall", "confusion_matrix", "c_matrix"]

_prob_based = ["roc_auc", "auc", "roc_curves"]


def _to_numpy(data):
    return data.cpu().numpy()


class Evaluation:
    def __init__(self, model=None, data_loader=None, device=_device, categories=None):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.categories = categories

    def _check_loader(self, data_loader):
        if data_loader is None:
            data_loader = self.data_loader

        if data_loader is None:
            raise Exception("No data loader is passed in, "
                            "you need to pass a data loader before make predictions.")
        return data_loader

    def _check_model(self, model):
        if model is None:
            model = self.model

        if model is None:
            raise Exception("No model is passed in, "
                            "you need to pass a model before make predictions")

    def predict(self, x):
        pred = predict(self.model.to(self.device), x.to(self.device))
        return pred

    def get_predictions(self, model=None, data_loader=None):
        model = self._check_model(model)
        data_loader = self._check_loader(data_loader)

        preds, targets = get_predictions(model, data_loader, self.device)
        return preds, targets

    def probable(self, x):
        prob = probable(self.model.to(self.device), x.to(self.device))
        return prob

    def get_probabilities(self, model=None, data_loader=None):
        model = self._check_model(model)
        data_loader = self._check_loader(data_loader)

        probs, targets = get_probabilities(model, data_loader, self.device)
        return probs, targets

    def evaluate(self, metric="accuracy", model=None, data_loader=None):
        """
        The main method in class Evaluation, to calculate multiple metric.

        :param metric: (str/list[str], default="accuracy") the metric method used. there are two kinds of metric methods
        and each has methods below.
        _pred_based = ["acc", "accuracy", "balanced_accuracy", "b_acc", "precision", "recall",
                       "confusion_matrix", "c_matrix"]
        _prob_based = ["roc_auc", "auc", "roc_curves"]
        :param model: (Model, default=None) the model to be used, if None, use model in __init__
        :param data_loader: (DataLoader, default=None) the data iter to be used, if None, use data_loader in __init__

        :return: (adarray/dict) the result
        """
        preds, targets = self.get_predictions(model, data_loader)
        probs, t_probs = self.get_probabilities(model, data_loader)
        preds = _to_numpy(preds)
        targets = _to_numpy(targets)
        probs = _to_numpy(probs)
        t_probs = _to_numpy(t_probs)

        out = None
        if isinstance(metric, list):
            out = dict()
            for key in metric:
                if key in _pred_based:
                    out[key] = metric_dict[key](preds, targets)
                if key in _prob_based:
                    out[key] = metric_dict[key](probs, t_probs)
            return out
        elif isinstance(metric, str):
            if metric in _pred_based:
                out = metric_dict[metric](preds, targets)
            if metric in _prob_based:
                out = metric_dict[metric](probs, t_probs)
            return out
        else:
            raise TypeError(f"the metric arg should be str or list of str, \
            but received {type(metric)}")

    def get_report(self, model=None, data_loader=None):
        preds, targets = self.get_predictions(model, data_loader)
        preds = _to_numpy(preds)
        targets = _to_numpy(targets)
        report = get_report(preds, targets, self.categories)
        return report


# class activation mapping(CAM)
def CAM(feature_of_conv, weight_of_classifier, class_idxs,
        size_upsample=(224, 224)):
    """ calculate the class activation mapping

    Args:
        feature_of_conv(tensor of shape[bs, c, h, w]): the output of the last Conv layer
            just before the GAP(global average pooling) and FC(fully connected)
        weight_of_classifier(tensor of shape[num_of_classes, c]): the weight of the FC
        class_idxs(int/list of int): the class index of the input image
        size_upsample(tuple): the output shape of CAM
    
    Returns:
        A list of 2d ndarray represents the CAM of the batch of inputs
    """
    bs, c, h, w = feature_of_conv.shape
    output_cams = []

    if type(class_idxs) == int:
        class_idxs = [class_idxs for _ in range(bs)]

    assert len(class_idxs) == bs, "the length of class_idxs not match the batch size."

    for i in range(bs):
        # compute cam
        weights = weight_of_classifier[class_idxs[i]].reshape((1, c))
        cam = weights @ feature_of_conv[i].reshape((c, h * w))
        cam = cam.reshape((h, w))

        # change to gray image
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())
        cam_img = np.uint8(255 * cam_img)
        output_cams.append(cv2.resize(cam_img, size_upsample))

    return output_cams
