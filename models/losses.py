import torch
import torchvision as tv
from torch import nn as nn
from torch.nn import functional as F

""" loss functions """


#todo make perceptual loss work


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg16 = tv.models.vgg16(pretrained=True).eval()
        layers = list(self.modules())
        layers[4].register_forward_hook(self._hook)
        layers[9].register_forward_hook(self._hook)
        layers[14].register_forward_hook(self._hook)
        layers[23].register_forward_hook(self._hook)
        self.update_target = False
        self.x_activations = []
        self.target_activations = []

    def _hook(self, module, input, activations):
        if self.update_target:
            self.target_activations.append(activations)
        else:
            self.x_activations.append(activations)

    def _loss(self, x, target):
        loss = 0.0
        for i, _ in enumerate(self.x_activations):
            x = self.x_activations[i]
            target = self.target_activations[i]
            loss += F.mse_loss(x, target, reduction='mean')
        return loss

    def forward(self, x, target):
        self.x_activations = []
        self.target_activations = []
        self.update_target = False
        self.vgg16(x)
        self.update_target = True
        self.vgg16(target)
        return self._loss(x, target)

#todo needs more work, does not work when 2 transforms are made to the image


class FlowfieldDiscountedLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, source, target, flowfield):
        loss = F.mse_loss(source, target, reduction='none')
        bad_bits = flowfield ** 2
        bad_bits[bad_bits <= 1.0] = 1.0
        bad_bits[bad_bits > 1.0] = 0
        mask = torch.prod(bad_bits, 3).expand(1, -1, -1, -1).permute(1, 0, 2, 3)
        loss = loss * mask
        return torch.sum(loss)


class DiscountBlackLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, source, target):
        """

        Ignores pixels with all channels set to zero
        Probably Not suitable for greyscale images

        :param source: image from network
        :param target: ground truth
        :return: the loss
        """
        loss = F.mse_loss(source, target, reduction='none')
        mask = torch.sum(target, dim=1, keepdim=True)
        mask[mask > 0.0] = 1.0
        mask[mask == 0.0] = 1e-6
        loss = loss * mask
        return torch.mean(loss), loss, mask