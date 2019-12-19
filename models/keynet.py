import torch
from models import knn


class KeyNet(knn.Container):
    def __init__(self, name, encoder,
                 keypoint, key2map,
                 decoder,
                 init_weights=True):
        super().__init__(name)
        self.encoder = encoder
        self.keypoint = keypoint
        self.ssm = knn.SpatialSoftmax()
        self.key2map = key2map
        self.decoder = decoder

        if init_weights:
            self._initialize_weights()

    def forward(self, x, x_t):

        z = self.encoder(x)

        heatmap = self.keypoint(x_t)
        k, p = self.ssm(heatmap, probs=True)
        m = self.key2map(k, height=z.size(2), width=z.size(3))

        x_t = self.decoder(torch.cat((z, m), dim=1))

        return x_t, z, k, m, p, heatmap

    def load(self, run_id, epoch):
        self.encoder.load(self.name, run_id, epoch)
        self.keypoint.load(self.name, run_id, epoch)
        self.decoder.load(self.name, run_id, epoch)

    def save(self, run_id, epoch):
        self.encoder.save(self.name, run_id, epoch)
        self.keypoint.save(self.name, run_id, epoch)
        self.decoder.save(self.name, run_id, epoch)

