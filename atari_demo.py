import gym
from utils import UniImageViewer, plot_keypoints_on_image
from datasets import pong_prepro, pong_grey_transform
from models import knn, transporter, vgg
from models import functional as KF
import torch.nn as nn
import config
import time

if __name__ == '__main__':

    v = UniImageViewer()
    env = gym.make('Pong-v0')
    s = env.reset()
    done = False

    args = config.config()

    nonlinearity, kwargs = nn.LeakyReLU, {"inplace": True}
    encoder_core = vgg.make_layers(vgg.vgg_cfg[args.model_type], nonlinearity=nonlinearity, nonlinearity_kwargs=kwargs)
    encoder = knn.Unit(args.model_in_channels, args.model_z_channels, encoder_core)
    decoder_core = vgg.make_layers(vgg.decoder_cfg[args.model_type])
    decoder = knn.Unit(args.model_z_channels, args.model_in_channels, decoder_core)
    keypoint_core = vgg.make_layers(vgg.vgg_cfg[args.model_type], nonlinearity=nonlinearity, nonlinearity_kwargs=kwargs)
    keypoint = knn.Unit(args.model_in_channels, args.model_keypoints, keypoint_core)
    keymapper = knn.GaussianLike(sigma=0.1)
    transporter_net = transporter.TransporterNet(encoder, keypoint, keymapper, decoder, init_weights=True,
                                                 combine_method=args.transporter_combine_mode)

    transporter_net = transporter_net.to(args.device)

    if args.load is not None:
        transporter_net.load(args.load)

    while not done:
        s, r, done, i = env.step(env.action_space.sample())
        s = pong_prepro(s)
        s = pong_grey_transform(s).unsqueeze(0).to(args.device)
        heatmap = transporter_net.keypoint(s)
        kp = KF.spacial_logsoftmax(heatmap)
        s = plot_keypoints_on_image(kp[0], s[0])
        v.render(s)
        time.sleep(0.04)