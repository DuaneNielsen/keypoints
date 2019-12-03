import torch
from torch import nn as nn
from torch.nn import Parameter, functional as F

from functional import gaussian_like_function


class Container(nn.Module):
    def __init__(self):
        super().__init__()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        pass


class GaussianLike(nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, kp, height, width):
        return gaussian_like_function(kp, height, width, self.sigma)


class CopyPoints(nn.Module):
    def __init__(self, height, width, sigma=0.1):
        super().__init__()
        self.height = height
        self.width = width

    def forward(self, kp):
        x, y = kp
        x = x.reshape(*x.shape, 1, 1).expand(*x.shape, self.height, self.width)
        y = y.reshape(*y.shape, 1, 1).expand(*y.shape, self.height, self.width)
        return torch.cat((x, y), dim=1)


class MaxPool2dA(nn.Module):
    def __init__(self, kernel_size, stride=0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, return_indices=True)
        self.indices = None

    def forward(self, x):
        x, self.indices = self.pool(x)
        return x


class MaxUnpool2dA(nn.Module):
    def __init__(self, pool, kernel_size, stride=0):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=kernel_size, stride=stride)
        self.pool = pool

    def forward(self, x):
        x = self.unpool(x, self.pool.indices)
        return x


class ActivationMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class SpatialSoftmax(torch.nn.Module):
    def __init__(self):
        super(SpatialSoftmax, self).__init__()

        # for backwards compatibility
        self.hs = Parameter(torch.zeros(1, 1, 14), requires_grad=False)
        self.ws = Parameter(torch.zeros(1, 1, 12), requires_grad=False)

    def marginalSoftMax(self, heatmap, dim):
        marginal = torch.sum(heatmap, dim=dim)
        sm = F.softmax(marginal, dim=2)
        return sm

    def forward(self, heatmap):
        height, width = heatmap.size(2), heatmap.size(3)
        h_sm, w_sm = self.marginalSoftMax(heatmap, dim=3), self.marginalSoftMax(heatmap, dim=2)
        hs = torch.linspace(0, 1, height).type_as(heatmap).expand(1, 1, -1).to(heatmap.device)
        ws = torch.linspace(0, 1, width).type_as(heatmap).expand(1, 1, -1).to(heatmap.device)
        h_k, w_k = torch.sum(h_sm * hs, dim=2, keepdim=True).squeeze(2), \
                   torch.sum(w_sm * ws, dim=2, keepdim=True).squeeze(2)
        return h_k, w_k