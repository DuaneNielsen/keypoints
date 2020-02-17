import torch
from keypoints.models import Container
import torch.nn as nn


class OutputBlock(nn.Module):
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


class Classifier(Container):
    def __init__(self, name, feature_block, encoder, output_block, init_weights=True):
        super().__init__()
        self.feature_block = feature_block
        self.encoder = encoder
        self.output_block = output_block

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        h = self.feature_block(x)
        h = self.encoder(h)
        return self.output_block(h)

    def load(self, run_id):
        self._load_block(self.feature_block, run_id, 'feature')
        self._load_block(self.encoder, run_id, 'encoder')
        self._load_block(self.output_block, run_id, 'output')

    def save(self, run_id):
        self._save_block(self.feature_block, run_id, 'feature')
        self._save_block(self.encoder, run_id, 'encoder')
        self._save_block(self.output_block, run_id, 'output')