import torch
from torch import nn as nn
from pathlib import Path
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


class KeyNet(nn.Module):
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

    def load_model(self, model_name, load_run_id):
        encoder_block_load_path = Path(f'data/keypoint_net/{model_name}/run{str(load_run_id)}/encoder.mdl')
        decoder_block_load_path = Path(f'data/keypoint_net/{model_name}/run{str(load_run_id)}/decoder.mdl')
        keypoint_block_load_path = Path(f'data/keypoint_net/{model_name}/run{str(load_run_id)}/keypoint.mdl')
        self.encoder.load_state_dict((torch.load(str(encoder_block_load_path))))
        self.decoder.load_state_dict(torch.load(str(decoder_block_load_path)))
        self.keypoint.load_state_dict(torch.load(str(keypoint_block_load_path)))

    def save_model(self, model_name, run_id):
        encoder_block_save_path = Path(f'data/keypoint_net/{model_name}/run{str(run_id)}/encoder.mdl')
        decoder_block_save_path = Path(f'data/keypoint_net/{model_name}/run{str(run_id)}/decoder.mdl')
        keypoint_block_save_path = Path(f'data/keypoint_net/{model_name}/run{str(run_id)}/keypoint.mdl')
        encoder_block_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.encoder.state_dict(), str(encoder_block_save_path))
        torch.save(self.decoder.state_dict(), str(decoder_block_save_path))
        torch.save(self.keypoint.state_dict(), str(keypoint_block_save_path))