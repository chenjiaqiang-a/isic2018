import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


eps = 1e-7
CLASS_NUM = 7


class SCELoss(nn.Module):
    def __init__(self, num_classes=CLASS_NUM, a=1, b=1):
        super(SCELoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.b = b
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        ce = self.cross_entropy(pred, labels)
        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        loss = self.a * ce + self.b * rce.mean()
        return loss


class RCELoss(nn.Module):
    def __init__(self, num_classes=CLASS_NUM, scale=1.0):
        super(RCELoss, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        loss = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * loss.mean()

class NRCELoss(nn.Module):
    def __init__(self, num_classes=CLASS_NUM, scale=1.0):
        super(NRCELoss, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        norm = 1 / 4 * (self.num_classes - 1)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * norm * rce.mean()


class NCELoss(nn.Module):
    def __init__(self, num_classes=CLASS_NUM, scale=1.0):
        super(NCELoss, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = -1 * torch.sum(label_one_hot * pred, dim=1) / (-pred.sum(dim=1))
        return self.scale * loss.mean()
    
    
    
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
    
    

class MAELoss(nn.Module):
    def __init__(self, num_classes=CLASS_NUM, scale=2.0):
        super(MAELoss, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = 1. - torch.sum(label_one_hot * pred, dim=1)
        return self.scale * loss.mean()

class NMAE(nn.Module):
    def __init__(self, num_classes=CLASS_NUM, scale=1.0):
        super(NMAE, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        norm = 1 / (self.num_classes - 1)
        loss = 1. - torch.sum(label_one_hot * pred, dim=1)
        return self.scale * norm * loss.mean()

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

class NGCELoss(nn.Module):
    def __init__(self, num_classes=CLASS_NUM, q=0.7, scale=1.0):
        super(NGCELoss, self).__init__()
        self.num_classes = num_classes
        self.q = q
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        numerators = 1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)
        denominators = self.num_classes - pred.pow(self.q).sum(dim=1)
        loss = numerators / denominators
        return self.scale * loss.mean()




class NCEandRCE(nn.Module):
    def __init__(self, alpha=1., beta=1., num_classes=CLASS_NUM):
        super(NCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.nce = NCELoss(num_classes=num_classes, scale=alpha)
        self.rce = RCELoss(num_classes=num_classes, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.rce(pred, labels)

class NCEandMAE(nn.Module):
    def __init__(self, alpha=1., beta=1., num_classes=CLASS_NUM):
        super(NCEandMAE, self).__init__()
        self.num_classes = num_classes
        self.nce = NCELoss(num_classes=num_classes, scale=alpha)
        self.mae = MAELoss(num_classes=num_classes, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.mae(pred, labels)

class NLNL(torch.nn.Module):
    def __init__(self, train_loader, num_classes=CLASS_NUM, ln_neg=1):
        super(NLNL, self).__init__()
        self.num_classes = num_classes
        self.ln_neg = ln_neg
        weight = torch.FloatTensor(num_classes).zero_() + 1.
        if not hasattr(train_loader.dataset, 'targets'):
            weight = [1] * num_classes
            weight = torch.FloatTensor(weight)
        else:
            for i in range(num_classes):
                weight[i] = (torch.from_numpy(np.array(train_loader.dataset.targets)) == i).sum()
            weight = 1 / (weight / weight.max())
        self.weight = weight.cuda()
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight)
        self.criterion_nll = torch.nn.NLLLoss()

    def forward(self, pred, labels):
        labels_neg = (labels.unsqueeze(-1).repeat(1, self.ln_neg)
                      + torch.LongTensor(len(labels), self.ln_neg).cuda().random_(1, self.num_classes)) % self.num_classes
        labels_neg = torch.autograd.Variable(labels_neg)

        assert labels_neg.max() <= self.num_classes-1
        assert labels_neg.min() >= 0
        assert (labels_neg != labels.unsqueeze(-1).repeat(1, self.ln_neg)).sum() == len(labels)*self.ln_neg

        s_neg = torch.log(torch.clamp(1. - F.softmax(pred, 1), min=1e-5, max=1.))
        s_neg *= self.weight[labels].unsqueeze(-1).expand(s_neg.size()).cuda()
        labels = labels * 0 - 100
        loss = self.criterion(pred, labels) * float((labels >= 0).sum())
        loss_neg = self.criterion_nll(s_neg.repeat(self.ln_neg, 1), labels_neg.t().contiguous().view(-1)) * float((labels_neg >= 0).sum())
        loss = ((loss+loss_neg) / (float((labels >= 0).sum())+float((labels_neg[:, 0] >= 0).sum())))
        return loss

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

class NormalizedFocalLoss(torch.nn.Module):
    def __init__(self, gamma=0.5, num_classes=CLASS_NUM, alpha=None, size_average=True, scale=1.0):
        super(NormalizedFocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, input, target):
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        normalizor = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = self.scale * loss / normalizor

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class NFLandRCE(torch.nn.Module):
    def __init__(self, alpha=1., beta=1., num_classes=CLASS_NUM, gamma=0.5):
        super(NFLandRCE, self).__init__()
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(gamma=gamma, num_classes=num_classes, scale=alpha)
        self.rce = RCELoss(num_classes=num_classes, scale=beta)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.rce(pred, labels)


class NFLandMAE(torch.nn.Module):
    def __init__(self, alpha=1., beta=1., num_classes=CLASS_NUM, gamma=0.5):
        super(NFLandMAE, self).__init__()
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(gamma=gamma, num_classes=num_classes, scale=alpha)
        self.mae = MAELoss(num_classes=num_classes, scale=beta)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.mae(pred, labels)


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