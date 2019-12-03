import torch
from torch import nn as nn

from knn import SpatialSoftmax


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