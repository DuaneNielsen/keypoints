import torch
from keypoints.models import knn
from keypoints import models as vgg


class KeyNet(knn.Container):
    def __init__(self, encoder,
                 keypoint, key2map,
                 decoder,
                 init_weights=True):
        super().__init__()
        self.encoder = encoder
        self.keypoint = keypoint
        self.ssm = knn.SpatialLogSoftmax()
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

    def load(self, directory):
        self.encoder.load(directory + '/encoder')
        self.keypoint.load(directory + '/keypoint')
        self.decoder.load(directory + '/decoder')

    def load_from_autoencoder(self, directory):
        self._initialize_weights()
        self.encoder.load(directory + '/encoder', out_block=False)
        self.keypoint.load(directory + '/encoder', in_block=True, core=True, out_block=False)
        self.decoder.load(directory + '/decoder', in_block=False, core=True, out_block=True)

    def save(self, directory):
        self.encoder.save(directory + '/encoder')
        self.keypoint.save(directory + '/keypoint')
        self.decoder.save(directory + '/decoder')


def make(args):
    nonlinearity, kwargs = nn.LeakyReLU, {"inplace": True}
    encoder_core = vgg.make_layers(vgg.vgg_cfg[args.model_type], nonlinearity=nonlinearity, nonlinearity_kwargs=kwargs)
    encoder = knn.Unit(args.model_in_channels, args.model_z_channels, encoder_core)
    decoder_core = vgg.make_layers(vgg.decoder_cfg[args.model_type])
    decoder = knn.Unit(args.model_z_channels + args.model_keypoints, args.model_in_channels, decoder_core)
    keypoint_core = vgg.make_layers(vgg.vgg_cfg[args.model_type], nonlinearity=nonlinearity, nonlinearity_kwargs=kwargs)
    keypoint = knn.Unit(args.model_in_channels, args.model_keypoints, keypoint_core)
    keymapper = knn.GaussianLike(sigma=0.1)
    kp_network = KeyNet(encoder, keypoint, keymapper, decoder, init_weights=True)

    if args.load is not None:
        kp_network.load(args.load)
    if args.transfer_load is not None:
        kp_network.load_from_autoencoder(args.transfer_load)

    return kp_network