import torch.nn as nn
from models.knn import ActivationMap
from math import floor

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


def make_layers(cfg, batch_norm=True, extra_in_channels=0,
                nonlinearity=None, nonlinearity_kwargs=None):
    nonlinearity_kwargs = {} if nonlinearity_kwargs is None else nonlinearity_kwargs
    nonlinearity = nn.ReLU(inplace=True) if nonlinearity is None else nonlinearity(**nonlinearity_kwargs)
    layers = []
    in_channels = cfg[0] + extra_in_channels
    for v in cfg[1:]:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'U':
            layers += [nn.UpsamplingBilinear2d(scale_factor=2)]
        elif v == 'L':
            layers += [ActivationMap()]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3)
            if batch_norm:
                layers += [nn.ReplicationPad2d(1), conv2d, nn.BatchNorm2d(v), nonlinearity]
            else:
                layers += [nn.ReplicationPad2d(1), conv2d, nonlinearity]
            in_channels = v
    return nn.Sequential(*layers)


"""
M -> MaxPooling
L -> Capture Activations for Perceptual loss
U -> Bilinear upsample
"""

decoder_cfg = {
    'A': [512, 512, 'U', 256, 256, 'U', 256, 256, 'U', 128, 'U', 64, 'U'],
    'F': [512, 512, 'U', 256, 256, 'U', 256, 256, 'U', 128, 64],
    'VGG_PONG': [32, 'U', 16, 'U', 16],
    'VGG_PONG_TRIVIAL': [16, 16],
}

vgg_cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'F': [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG_PONG': [16, 'M', 16, 'M', 32],
    'VGG_PONG_TRIVIAL': [16, 16],
}
