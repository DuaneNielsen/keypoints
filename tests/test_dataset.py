import numpy as np
import cv2
import matplotlib.pyplot as plt

import cma_es
from keypoints.ds import datasets as d
from keypoints.ds.datasets import Pos, SquareDataset, AtariDataset, if_done_or_nonzero_reward
from torch.utils.data import DataLoader
import gym
from utils import UniImageViewer
from torchvision import transforms as T
import time

def test_render_square():
    a3 = d.square(Pos(40, 40), Pos(50, 50))
    im = np.zeros([128, 128, 3], dtype=np.uint8)
    cv2.fillPoly(im, a3, (0, 255, 255))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.imshow(im)
    plt.show()


def test_dataset():
    ds = SquareDataset(size=10)
    loader = DataLoader(ds, batch_size=10)

    for img in loader:
        plt.imshow(img[0])
        plt.show()


def test_pong():
    v = UniImageViewer()
    l = UniImageViewer(title='processed', screen_resolution=(32, 32))
    env = gym.make('Pong-v0')

    s = env.reset()
    done = False

    while not done:
        s, r, done, info = env.step(cma_es.sample())
        v.render(s)
        s = d.pong_color_prepro(s)
        #s = cv2.cvtColor(s, cv2.COLOR_RGB2GRAY)
        #s = s[34:168, :]
        #s = cv2.resize(s, dsize=(32, 32), interpolation=cv2.INTER_AREA)
        l.render(s)


def test_pong_fill():
    l = 6000
    display = True

    ds = AtariDataset('Pong-v0', l, d.pong_prepro,
                      transforms = T.Compose([T.ToTensor(), T.ToPILImage()]),
                      end_trajectory=if_done_or_nonzero_reward)
    assert len(ds) >= l
    print(len(ds))

    disp = UniImageViewer(screen_resolution=(512, 512))

    for img1, img2 in ds:
        #i = np.concatenate((img[0], img[1]), axis=1)
        if display:
            disp.render(img1)
            time.sleep(0.03)
        else:
            plt.imshow(img1, cmap='gray', vmin=0, vmax=256.0)
            plt.show()


def plot_heatmap2d(z, title):
    # show height map in 2d
    plt.figure()
    plt.title(title)
    p = plt.imshow(z.detach().numpy())
    plt.colorbar(p)
    plt.show()


def test_mapper():
    ds = d.MapperDataset(16, 16, 3, 10)
    x, y = ds[0]
    plot_heatmap2d(x[0], 'point function')
    plot_heatmap2d(y[0], 'target function')
