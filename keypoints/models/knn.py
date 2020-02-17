from pathlib import Path

import torch
from torch import nn as nn
from keypoints.models.functional import gaussian_like_function, spacial_softmax, spacial_logsoftmax


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

    def forward(self, *input):
        raise NotImplementedError()

    def save(self, directory):
        raise NotImplementedError()

    def load(self, directory):
        raise NotImplementedError()


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
        super().__init__()

    def forward(self, heatmap, probs=False):
        return spacial_softmax(heatmap, probs)

class SpatialLogSoftmax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, heatmap, probs=False):
        return spacial_logsoftmax(heatmap, probs)


class Unit(nn.Module):
    def __init__(self, in_channels, out_channels, core, batch_norm=True):
        super().__init__()
        core_in_channels, core_out_channels = self._core_channels(core)

        in_block = [nn.ReplicationPad2d(1), nn.Conv2d(in_channels, core_in_channels, kernel_size=3, stride=1)]
        if batch_norm:
            in_block += [nn.BatchNorm2d(core_in_channels)]
        in_block += [nn.LeakyReLU(inplace=True)]
        self.in_block = nn.Sequential(*in_block)

        self.core = core

        out_block = [nn.Conv2d(core_out_channels, out_channels, kernel_size=1, stride=1)]
        out_block += [nn.LeakyReLU(inplace=True)]
        self.out_block = nn.Sequential(*out_block)

    def forward(self, x):
        h = self.in_block(x)
        h = self.core(h)
        return self.out_block(h)

    def _core_channels(self, core):
        """guess the number of channels in and out of core"""
        first = None
        last = None
        for m in core.modules():
            if isinstance(m, nn.Conv2d):
                if first is None:
                    first = m
                last = m
        return first.in_channels, last.out_channels

    def _save_block(self, directory, unit, block):
        path = Path(f'{directory}/{unit}.mdl')
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(block.state_dict(), str(path))

    def _load_block(self, directory, unit, block, map_device=None):
        path = Path(f'{directory}/{unit}.mdl')
        if map_device:
            f = torch.load(str(path), map_device)
            block.load_state_dict(f)
        else:
            block.load_state_dict(torch.load(str(path)))

    def save(self, directory):
        self._save_block(directory, 'in_block', self.in_block)
        self._save_block(directory, 'core', self.core)
        self._save_block(directory, 'out_block', self.out_block)

    def load(self, directory, in_block=True, core=True, out_block=True, map_device=None):
        if in_block:
            self._load_block(directory, 'in_block', self.in_block, map_device=map_device)
        if core:
            self._load_block(directory, 'core', self.core, map_device=map_device)
        if out_block:
            self._load_block(directory, 'out_block', self.out_block, map_device=map_device)


class FeatureBlock(nn.Module):
    def __init__(self, channels, out_channels=64, batch_norm=True):
        super().__init__()
        block = [nn.Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1)]
        if batch_norm:
            block += [nn.BatchNorm2d(out_channels)]
        block += [nn.ReLU()]
        self.feature_block = nn.Sequential(*block)

    def forward(self, x):
        return self.feature_block(x)


class OutputBlock(nn.Module):
    def __init__(self, out_channels, batch_norm=True):
        super().__init__()
        block = [nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)]
        if batch_norm:
            block += [nn.BatchNorm2d(out_channels)]
        block += [nn.ReLU()]
        self.output_block = nn.Sequential(*block)

    def forward(self, x):
        return self.output_block(x)


class Coords(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """ adds 2 channels that carry co-ordinate information """
        b, h, w = x.size(0), x.size(2), x.size(3)
        hm = torch.linspace(0, 1, h, dtype=x.dtype, device=x.device).reshape(1, 1, h, 1).repeat(b, 1, 1, w)
        wm = torch.linspace(0, 1, w, dtype=x.dtype, device=x.device).reshape(1, 1, 1, w).repeat(b, 1, h, 1)
        return torch.cat((x, hm, wm), dim=1)


class Transporter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, phi_xs, heta_xs, phi_xt, heta_xt):
        return phi_xs * (1 - heta_xs) * (1 - heta_xt) + phi_xs * heta_xt


