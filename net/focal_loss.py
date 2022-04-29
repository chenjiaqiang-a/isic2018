import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, num_classes: int, alpha: list, gamma=2, reduction='mean'):
        """ Focal Loss

        Args:
            num_classes (int): 类别数量 number of classes
            alpha (list): 类别权重 class weight
            gamma (int/float): 难易样本调节参数 focusing parameter
            reduction (string): 'mean', 'sum', 'none'
        """
        super(FocalLoss, self).__init__()

        assert len(alpha) == num_classes, "alpha size doesn't match with class number"
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds, labels):
        """
        Shape:
            preds: [B, N, C] or [B, C]
            labels: [B, N] or [B]
        """
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1)
        preds_logsoft = torch.log(preds_softmax)

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        # torch.pow((1-preds_softmax), self.gamma) - (1-pt)**γ
        loss = - torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        loss = torch.mul(self.alpha, loss.t())

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

    def __str__(self) -> str:
        return "Criterion: Focal Loss\n α = {}\n γ = {}".format(self.alpha, self.gamma)
