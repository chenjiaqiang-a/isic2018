import torch
import torch.nn as nn

from .attack import Attack


class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    """

    def __init__(self, model, eps=0.007, loss=nn.CrossEntropyLoss()):
        """
        Args:
            model (nn.Module): model to attack.
            eps (float): maximum perturbation. (Default: 0.007)
            loss (nn): loss function. (Default: nn.CrossEntropyLoss())
        """
        super().__init__("FGSM", model)
        self.eps = eps
        self.loss = loss

    def forward(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        images.requires_grad = True
        outputs = self.model(images)

        # Calculate loss
        cost = self.loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(
            cost, images, retain_graph=False, create_graph=False)[0]

        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
