import torch
from models import knn


class TransporterNet(knn.Container):
    def __init__(self,
                 feature_cnn,
                 keypoint_cnn,
                 key2map,
                 decoder,
                 init_weights=True):
        super().__init__()
        self.feature = feature_cnn
        self.keypoint = keypoint_cnn
        self.ssm = knn.SpatialLogSoftmax()
        self.key2map = key2map
        self.decoder = decoder

        if init_weights:
            self._initialize_weights()

    def forward(self, xs, xt):

        with torch.no_grad():
            phi_xs = self.feature(xs)
            heatmap_xs = self.keypoint(xs)
            k_xs, p_xs = self.ssm(heatmap_xs, probs=True)
            heta_xs = self.key2map(k_xs, height=phi_xs.size(2), width=phi_xs.size(3))

        phi_xt = self.feature(xt)
        heatmap_xt = self.keypoint(xt)
        k_xt, p_xt = self.ssm(heatmap_xt, probs=True)
        heta_xt = self.key2map(k_xt, height=phi_xt.size(2), width=phi_xt.size(3))

        z = phi_xs * (1 - heta_xs) * (1 - heta_xt) + phi_xt * heta_xt

        x_t = self.decoder(z)

        return x_t, z, k_xt, heta_xt, p_xt, heatmap_xt

    def load(self, directory):
        self.feature.load(directory + '/encoder')
        self.keypoint.load(directory + '/keypoint')
        self.decoder.load(directory + '/decoder')

    def load_from_autoencoder(self, directory):
        self._initialize_weights()
        self.feature.load(directory + '/encoder', out_block=False)
        self.keypoint.load(directory + '/encoder', in_block=True, core=True, out_block=False)
        self.decoder.load(directory + '/decoder', in_block=False, core=True, out_block=True)

    def save(self, directory):
        self.feature.save(directory + '/encoder')
        self.keypoint.save(directory + '/keypoint')
        self.decoder.save(directory + '/decoder')

