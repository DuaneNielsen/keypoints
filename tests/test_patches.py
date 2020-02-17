import cma_es
import keypoints.ds.datasets
import torch
import gym
import gym_wrappers
import main
from keypoints.models import transporter
import config
from matplotlib import pyplot as plt
from keypoints.models import patch_axis

def test_patch():

    args = config.config(['--config', '../configs/cma_es/exp2/baseline.yaml'])

    torch.manual_seed(0)
    datapack = keypoints.ds.datasets.datasets[args.dataset]
    env = gym.make(datapack.env)
    env = gym_wrappers.RewardCountLimit(env, 5)
    done = False
    env.reset()
    transporter_net = transporter.make(args, map_device='cpu')
    view = main.Keypoints(transporter_net)

    while not done:
        s, r, done, info = env.step(cma_es.sample())
        s = datapack.prepro(s)
        s_t = datapack.transforms(s).unsqueeze(0)
        kp = view(s_t)
        print(kp)
        env.render()


def test_patch():
    x, y = patch_axis(0.3, 10, 1)
    plt.plot(x, y)
    plt.show()