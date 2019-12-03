import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

from models.autoencoder import VGGAutoEncoder
from knn import Container, GaussianLike, ActivationMap, Identity
from models.keypoints import Keypoint, VGGKeypoints

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


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


class VGGBlock(nn.Module):

    def __init__(self, features):
        super().__init__()
        self.features = features

    def forward(self, x):
        x = self.features(x)
        return x


class VGG(Container):
    def __init__(self, feature_block, output_block, init_weights=True):
        super().__init__()

        if init_weights:
            self._initialize_weights()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = cfg[0]
    for v in cfg[1:]:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'U':
            layers += [nn.UpsamplingBilinear2d(scale_factor=2)]
        elif v == 'L':
            layers += [ActivationMap()]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)





"""
M -> MaxPooling
L -> Capture Activations for Perceptual loss
"""

auto_cfgs = {
    'A': {"encoder": [3, 64, 'M', 128, 'M', 256, 256, 'M', 512, 512],
          "decoder": [512, 256, 'U', 256, 128, 'U', 128, 64, 'U', 64, 32, 32, 3]},
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, output_block, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), output_block, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def _vgg_auto(cfg, pretrained, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    encoder = make_layers(auto_cfgs[cfg]["encoder"], batch_norm=True)
    decoder = make_layers(auto_cfgs[cfg]["decoder"], batch_norm=True)
    return VGGAutoEncoder(encoder, decoder)


def _vgg_kp(cfg, pretrained, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    # need to add the number of keypoints to the channels of 1st layer
    encoder = make_layers(auto_cfgs[cfg]["encoder"], batch_norm=True)
    decoder_cfg = auto_cfgs[cfg]["decoder"].copy()
    decoder_cfg[0] += kwargs["num_keypoints"]
    decoder = make_layers(decoder_cfg, batch_norm=True)
    kp_encoder = make_layers(auto_cfgs[cfg]["encoder"], batch_norm=True)
    keypoints = Keypoint(kp_encoder, num_keypoints=kwargs['num_keypoints'])
    keymapper = GaussianLike(sigma=kwargs["sigma"])
    #keymapper = CopyPoints(height=kwargs["height"], width=kwargs["width"], sigma=kwargs["sigma"])
    return VGGKeypoints(encoder, decoder, keypoints, keymapper, init_weights=True)


def _vgg_kp_test(cfg, pretrained, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    # need to add the number of keypoints to the channels of 1st layer
    auto_cfgs[cfg]["decoder"][0] += kwargs["num_keypoints"]
    encoder = make_layers(auto_cfgs[cfg]["encoder"], batch_norm=True)
    decoder = make_layers(auto_cfgs[cfg]["decoder"], batch_norm=True)
    keypoints = make_layers(auto_cfgs[cfg]["encoder"], batch_norm=True)
    keymapper = Identity()
    return VGGKeypoints(encoder, decoder, keypoints, keymapper)


def vgg11(output_block, pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', output_block, False, pretrained, progress, **kwargs)


def vgg11_bn_auto():
    return _vgg_auto('A', pretrained=False)


def vgg11_bn_keypoint(num_keypoints=10, sigma=1.0, **kwargs):
    return _vgg_kp('A', pretrained=False, num_keypoints=num_keypoints, sigma=sigma, **kwargs)


def vgg11_bn_keypoint_test(height, width, num_keypoints=10):
    return _vgg_kp_test('A', pretrained=False, height=height, width=width, num_keypoints=num_keypoints)


def vgg11_bn(output_block, pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', output_block, True, pretrained, progress, **kwargs)



def vgg13(output_block, pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', output_block, False, pretrained, progress, **kwargs)



def vgg13_bn(output_block, pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', output_block, True, pretrained, progress, **kwargs)



def vgg16(output_block, pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', output_block, False, pretrained, progress, **kwargs)

def vgg16_bn_auto():
    return _vgg_auto('D', pretrained=False)


def vgg16_bn(output_block, pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', output_block, True, pretrained, progress, **kwargs)



def vgg19(output_block, pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', output_block, False, pretrained, progress, **kwargs)



def vgg19_bn(output_block, pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', output_block, True, pretrained, progress, **kwargs)