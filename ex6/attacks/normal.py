import torch

from .attack import Attack


class Normal(Attack):
    r"""
    Normal perturbations 常规扰动 （加入高斯随机噪声）

    """
    def __init__(self, model, eps=8/255):
        """
        Args:
            model (nn.Module): model to attack.
            eps (float): maximum perturbation. (Default: 8/255).
        """
        super().__init__("Normal", model)
        self.eps = eps

    def forward(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # Random perturbations
        adv_images = images + torch.randn_like(images).uniform_(-self.eps, self.eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
