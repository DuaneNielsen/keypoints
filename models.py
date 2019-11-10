import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
import torchvision as tv
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

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


class VGGClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


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


class VGG(Container):
    def __init__(self, feature_block, output_block, init_weights=True):
        super().__init__()

        if init_weights:
            self._initialize_weights()


class VGGAutoEncoder(Container):
    def __init__(self, encoder, decoder, init_weights=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return z, x


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


def squared_diff(h, height):
    hs = torch.linspace(0, 1, height, device=h.device).type_as(h).expand(h.shape[0], h.shape[1], height)
    hm = h.expand(height, -1, -1).permute(1, 2, 0)
    hm = ((hs - hm) ** 2)
    #hm = (hs - hm).abs()
    return hm


def gaussian_like_function(kp, height, width, sigma=0.1):
    h, w = kp
    hm = squared_diff(h, height)
    wm = squared_diff(w, width)
    hm = hm.expand(width, -1, -1, -1).permute(1, 2, 3, 0)
    wm = wm.expand(height, -1, -1, -1).permute(1, 2, 0, 3)
    gm = - (hm + wm + 1e-6).sqrt_() / (2 * sigma ** 2)
    #gm = - (hm + wm) / (2 * sigma ** 2)
    gm = torch.exp(gm)
    return gm


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


class Keypoint(nn.Module):
    def __init__(self, encoder, num_keypoints):
        super().__init__()
        self.encoder = encoder
        self.reducer = nn.Sequential(nn.Conv2d(512, num_keypoints, kernel_size=1, stride=1),
                                     nn.BatchNorm2d(num_keypoints),
                                     nn.ReLU())
        self.ssm = SpatialSoftmax()

    def forward(self, x_t):
        z_t = self.encoder(x_t)
        z_t = self.reducer(z_t)
        k = self.ssm(z_t)
        return k


class VGGKeypoints(nn.Module):
    def __init__(self, encoder, decoder, keypoint, keymapper, init_weights=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.keypoint = keypoint
        self.keymapper = keymapper
        if init_weights:
            self._initialize_weights()

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

    def forward(self, x, x_t):
        z = self.encoder(x)
        k = self.keypoint(x_t)
        gm = self.keymapper(k, height=z.size(2), width=z.size(3))
        x_t = self.decoder(torch.cat((z, gm), dim=1))
        return x_t, z, k


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


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


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