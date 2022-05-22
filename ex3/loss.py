import torch
import torch.nn as nn
import torch.nn.functional as F



class FocalLoss(nn.Module):
    def __init__(self, alpha: list = None, gamma=2, num_classes: int = 2, reduction='mean'):
        """ Focal Loss

        Args:
            alpha (list): 类别权重 class weight
            gamma (int/float): 难易样本调节参数 focusing parameter
            num_classes (int): 类别数量 number of classes
            reduction (string): 'mean', 'sum', 'none'
        """
        super(FocalLoss, self).__init__()
        if alpha == None:
            self.alpha = torch.Tensor([1 for _ in range(num_classes)])
        else:
            assert len(alpha) == num_classes, "alpha size doesn't match with class number"
            self.alpha = torch.Tensor(alpha)
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
        loss = - torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft) # torch.pow((1-preds_softmax), self.gamma) - (1-pt)**γ
        loss = torch.mul(self.alpha, loss.t())
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

    def __str__(self) -> str:
        return "FocalLoss(α={}, γ={})".format(self.alpha, self.gamma)


class DiceLoss(nn.Module):
    def __init__(self, weight = None):
        """ Dice Loss
            dice_loss = 1 - 2*p*t / (p^2 + t^2). p and t represent predict and target.

        Args:
            weight: class weights. An array of shape [C,]
        """
        super(DiceLoss, self).__init__()
        if weight is not None:
            self.weight = torch.Tensor(weight)
        self.smooth = 1e-5

    def forward(self, predict, target):
        """
        Shape:
            predict: [N, C, H, W]
            target: [N, H, W]
        """
        N, C = predict.size()[:2]
        predict = predict.view(N, C, -1) # (N, C, *)
        target = target.view(N, 1, -1) # (N, 1, *)

        predict = F.softmax(predict, dim=1) # (N, C, *) ==> (N, C, *)
        ## convert target(N, 1, *) into one hot vector (N, C, *)
        target_onehot = torch.zeros(predict.size()).cuda()  # (N, 1, *) ==> (N, C, *)
        target_onehot.scatter_(1, target, 1)  # (N, C, *)

        intersection = torch.sum(predict * target_onehot, dim=2)  # (N, C)
        union = torch.sum(predict.pow(2), dim=2) + torch.sum(target_onehot, dim=2)  # (N, C)
        ## p^2 + t^2 >= 2*p*t, target_onehot^2 == target_onehot
        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)  # (N, C)

        if hasattr(self, 'weight'):
            if self.weight.type() != predict.type():
                self.weight = self.weight.type_as(predict)
                dice_coef = dice_coef * self.weight * C  # (N, C)
        dice_loss = 1 - torch.mean(dice_coef)  # 1

        return dice_loss

    def __str__(self) -> str:
        if hasattr(self, 'weight'):
            return "DiceLoss(weight={})".format(self.weight)
        else:
            return "DiceLoss()"



class DiceCE(nn.Module):
    def __init__(self, lamb_dice = 1., lamb_ce = 1., weight: list = None):
        """ Dice-CrossEntropy Loss 

        Args:
            lamb_dice (float): factor of Dice Loss
            lamb_ce (float): factor of Cross Entropy Loss
            weight (list): class weight
        """
        super(DiceCE, self).__init__()
        self.lamb_dice = lamb_dice
        self.lamb_ce = lamb_ce
        self.dice = DiceLoss(weight=weight)
        if weight == None:
            self.ce = nn.CrossEntropyLoss()
        else:
            self.ce = nn.CrossEntropyLoss(weight=torch.tensor(weight))

    def forward(self, pred, labels):
        return self.lamb_dice * self.dice(pred, labels) + self.lamb_ce * self.ce(pred, labels)
    
    def __str__(self) -> str:
        return "DiceCELoss(λ_dice={}, λ_ce={})\n{}\n{}".format(self.lamb_dice, self.lamb_ce, self.dice, self.ce)


class DiceFocal(nn.Module):
    def __init__(self, lamb_dice = 1., lamb_focal = 1., weight: list = None):
        """ DiceFocal Loss

        Args:
            lamb_dice (float): factor of Dice Loss
            lamb_focal (float): factor of Focal Loss
            weight (list): class weight
        """
        super(DiceFocal, self).__init__()
        self.lamb_dice = lamb_dice
        self.lamb_focal = lamb_focal
        self.dice = DiceLoss(weight=weight)
        self.focal = FocalLoss(alpha=weight)

    def forward(self, pred, labels):
        return self.lamb_dice * self.dice(pred, labels) + self.lamb_focal * self.focal(pred, labels)
    
    def __str__(self) -> str:
        return "DiceFocalLoss(λ_dice={}, λ_focal={})\n{}\n{}".format(self.lamb_dice, self.lamb_focal, self.dice, self.focal)