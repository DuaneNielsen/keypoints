import numpy as np
import cv2
import matplotlib.pyplot as plt
import datasets as b
from datasets import Pos, SquareDataset, AtariDataset, pong_prepro
from torch.utils.data import DataLoader
import gym
from utils import UniImageViewer

def test_render_square():
    a3 = b.square(Pos(40, 40), Pos(50, 50))
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
    l = UniImageViewer(title='grey', screen_resolution=(32, 32))
    env = gym.make('Pong-v0')

    s = env.reset()
    done = False

    while not done:
        s, r, done, info = env.step(env.action_space.sample())
        v.render(s)
        s = cv2.cvtColor(s, cv2.COLOR_RGB2GRAY)
        s = s[25:, :]
        s = cv2.resize(s, dsize=(32, 32), interpolation=cv2.INTER_LINEAR)
        l.render(s)


def test_pong_fill():
    ds = AtariDataset('Pong-v0', 3000, pong_prepro)
    assert len(ds) >= 3000
    print(len(ds))