from pathlib import Path

import torch
from torch import nn as nn
from models.functional import gaussian_like_function, spacial_softmax


class Container(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

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

    def _save_block(self, block, run_id, blockname):
        path = Path(f'data/models/{self.name}/run_{run_id}/{blockname}.mdl')
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(block.state_dict(), str(path))

    def _load_block(self, block, run_id, blockname):
        path = Path(f'data/models/{self.name}/run_{run_id}/{blockname}.mdl')
        block.load_state_dict(torch.load(str(path)))

    def forward(self, *input):
        raise NotImplementedError()

    def save(self, run_id, epoch):
        raise NotImplementedError()

    def load(self, run_id, epoch):
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


class Unit(nn.Module):
    def __init__(self, name, in_channels, out_channels, core, in_batch_norm=True, out_batch_norm=True):
        super().__init__()
        self.name = name
        core_in_channels, core_out_channels = self._core_channels(core)

        in_block = [nn.ReplicationPad2d(1), nn.Conv2d(in_channels, core_in_channels, kernel_size=3, stride=1)]
        if in_batch_norm:
            in_block += [nn.BatchNorm2d(core_in_channels)]
        in_block += [nn.LeakyReLU(inplace=True)]
        self.in_block = nn.Sequential(*in_block)

        self.core = core

        out_block = [nn.Conv2d(core_out_channels, out_channels, kernel_size=1, stride=1)]
        if out_batch_norm:
            out_block += [nn.BatchNorm2d(out_channels)]
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

    def _save_block(self, model_name, run_id, epoch, unit, block):
        path = Path(f'data/models/{model_name}/run_{run_id}/{epoch}/{self.name}/{unit}.mdl')
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(block.state_dict(), str(path))

    def _load_block(self, model_name, run_id, epoch, unit, block):
        path = Path(f'data/models/{model_name}/run_{run_id}/{epoch}/{self.name}/{unit}.mdl')
        block.load_state_dict(torch.load(str(path)))

    def save(self, model_name, run_id, epoch):
        self._save_block(model_name, run_id, epoch, 'in_block', self.in_block)
        self._save_block(model_name, run_id, epoch, 'core', self.core)
        self._save_block(model_name, run_id, epoch, 'out_block', self.out_block)

    def load(self, model_name, run_id, epoch):
        self._load_block(model_name, run_id, epoch, 'in_block', self.in_block)
        self._load_block(model_name, run_id, epoch, 'core', self.core)
        self._load_block(model_name, run_id, epoch, 'out_block', self.out_block)


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