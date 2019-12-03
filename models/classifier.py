import torch
from pathlib import Path
from knn import Container
import torch.nn as nn


class FeatureBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Conv2d(channels, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.nl = nn.ReLU()

    def forward(self, x):
        return self.nl(self.bn(self.block(x)))


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
    def __init__(self, feature_block, middle_block, output_block, init_weights=True):
        super().__init__()
        self.feature_block = feature_block
        self.middle_block = middle_block
        self.output_block = output_block

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        h = self.feature_block(x)
        h = self.middle_block(h)
        return self.output_block(h)

    def _load_block(self, block, run_id, blockname):
        path = Path(f'data/classifier/models/run_{run_id}/{blockname}.mdl')
        block.load_state_dict(torch.load(str(path)))

    def _save_block(self, block, run_id, blockname):
        path = Path(f'data/classifier/models/run_{run_id}/{blockname}.mdl')
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(block.state_dict(), str(path))

    def load(self, run_id):
        self._load_block(self.feature_block, run_id, 'feature')
        self._load_block(self.middle_block, run_id, 'middle')
        self._load_block(self.output_block, run_id, 'output')

    def save(self, run_id):
        self._save_block(self.feature_block, run_id, 'feature')
        self._save_block(self.middle_block, run_id, 'middle')
        self._save_block(self.output_block, run_id, 'output')