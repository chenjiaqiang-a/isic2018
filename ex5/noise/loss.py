import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


eps = 1e-7
CLASS_NUM = 7


    
class WeightedCELoss(nn.Module):
    def __init__(self, num_classes=CLASS_NUM, reduction='mean'):
        super(WeightedCELoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, labels: torch.Tensor, sample_weight: torch.Tensor):
        """w * y * log(p)

        Args:
            pred (torch.Tensor): 网络输出值
            labels (torch.Tensor): 标签
            sample_weight (torch.Tensor): 逐样本权重

        Shape:
            pred: N, C
            label: N
            sample_weight: N

        Returns:
            tensor.float: 逐样本赋权的损失值
        """

        pred = F.log_softmax(pred, dim=1)                                           # N, C
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device) # C -> N, C
        loss = label_one_hot * pred                                                 # N, C
        # 赋权
        loss = - 1 * torch.sum(label_one_hot * pred, dim=1) * sample_weight          # N
        
        # 规约
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        else:
            raise ValueError('Unsupported reduction type.')

        return loss
    
    


class GCELoss(nn.Module):
    def __init__(self, num_classes=CLASS_NUM, q=0.7):
        super(GCELoss, self).__init__()
        self.q = q
        self.num_classes = num_classes

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean()



class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0.5, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * torch.autograd.Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class pNorm(nn.Module):
    def __init__(self, p=0.5):
        super(pNorm, self).__init__()
        self.p = p

    def forward(self, pred, p=None):
        if p:
            self.p = p
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1)
        norm = torch.sum(pred ** self.p, dim=1)
        return norm.mean()


class SR(nn.Module):
    def __init__(self, loss, tau, p, lamb) -> None:
        super(SR, self).__init__()
        self.loss = loss
        self.pnorm = pNorm(p)
        self.tau = tau
        self.lamb = lamb

    def forward(self, outputs, labels, p=None):
        return self.loss(outputs / self.tau, labels) + self.lamb * self.pnorm(outputs / self.tau, p)