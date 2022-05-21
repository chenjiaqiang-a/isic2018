import torch
import torch.nn as nn

from .attack import Attack


class PGD(Attack):
    r"""
    PGD attack in 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]
    
    Support random start in 'Fast Is Better than Free: Revisiting Adversarial Training'
    [http://arxiv.org/abs/2001.03994]


    Distance Measure : Linf

    """

    def __init__(self, model, eps=0.3, alpha=2/255, steps=40,
                 random_start=True, loss=nn.CrossEntropyLoss()):
        """
        Args:
            model (nn.Module): model to attack.
            eps (float): maximum perturbation. (Default: 0.3)
            alpha (float): step size. (Default: 2/255)
            steps (int): number of steps. (Default: 40)
            random_start (bool): using random initialization of perturbations. (Default: True)
        """
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.loss = loss

    def forward(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + \
                torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            # Calculate loss
            cost = self.loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            # clip the perturbations
            delta = torch.clamp(adv_images - images,
                                min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
