import gym

import cma_es
from utils import UniImageViewer, plot_keypoints_on_image
from keypoints.ds import datasets as ds
from keypoints.models import transporter, functional as KF
import torch
import config
import time
from torchvision.transforms import functional as TVF

if __name__ == '__main__':

    args = config.config()

    with torch.no_grad():
        v = UniImageViewer()

        datapack = ds.datasets[args.dataset]
        transporter_net = transporter.make(args).to(args.device)

        if args.load is not None:
            transporter_net.load(args.load)

        env = gym.make(datapack.env)

        while True:
            s = env.reset()
            done = False

            while not done:
                s, r, done, i = env.step(cma_es.sample())
                s = datapack.prepro(s)
                s_t = datapack.transforms(s).unsqueeze(0).to(args.device)
                heatmap = transporter_net.keypoint(s_t)
                kp = KF.spacial_logsoftmax(heatmap)
                s = TVF.to_tensor(s).unsqueeze(0)
                s = plot_keypoints_on_image(kp[0], s[0])
                v.render(s)
                time.sleep(0.04)