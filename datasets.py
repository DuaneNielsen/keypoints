import collections
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.utils.data
import torchvision as tv
from torchvision import transforms
import gym

Pos = collections.namedtuple('Pos', 'x, y')


class ConfigException(Exception):
    pass


""" dataset which generates a white square in the middle of color image """

def square(offset, height):
    top_left = Pos(offset.x, offset.y)
    top_right = Pos(offset.x + height.x, offset.y)
    bottom_left = Pos(offset.x, offset.y + height.y)
    bottom_right = Pos(offset.x + height.x, offset.y + height.y)
    return np.array(
        [[[top_left], [top_right], [bottom_right], [bottom_left]]],
        dtype=np.int32)


class Square():
    def __init__(self, offset, height):
        self.offset = offset
        self.height = height

    def __call__(self):
        return square(self.offset, self.height)


def image(poly, screensize, color=(255, 255, 255)):
    im = np.zeros([*screensize, 3], dtype=np.uint8)
    im = cv2.fillPoly(im, poly(), color)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


class SquareDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, size, transform=None):
        super().__init__()
        self.size = size
        self.poly = Square(Pos(40, 40), Pos(50, 50))
        self.transform = transform

    def __getitem__(self, item):
        img = image(self.poly, screensize=(128, 128), color=(255, 255, 255))
        if self.transform:
            img = self.transform(img)
        return img, img

    def __len__(self):
        return self.size


""" Atari dataset generator """

def pong_prepro(s):
    s = cv2.cvtColor(s, cv2.COLOR_RGB2GRAY)
    s = s[25:, :]
    s = cv2.resize(s, dsize=(32, 32), interpolation=cv2.INTER_LINEAR)
    return s


class AtariDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, env, size, prepro):
        super().__init__()
        self.env = gym.make(env)
        self.prepro = prepro
        self.trajectories = []
        self.index = []
        self._fill(size)

    def _trajectory(self):
        frames = []
        s = self.env.reset()
        ps = self.prepro(s)
        frames.append(ps)
        done = False

        while not done:
            s, r, done, info = self.env.step(self.env.action_space.sample())
            ps = self.prepro(s)
            frames.append(ps)

        return np.stack(frames, axis=0)

    def __len__(self):
        return len(self.index)

    def _fill(self, samples):
        traj_id = 0
        while len(self) < samples:
            t = self._trajectory()
            self.trajectories.append(t)
            for i in range(t.shape[0] - 1):
                self.index.append((traj_id, i))
            traj_id += 1

    def __getitem__(self, item):
        trajectory, i = self.index[item]
        return self.trajectories[trajectory][i], self.trajectories[trajectory][i + 1]


def get_dataset(data_root, dataset, run_type):
    size = {'full': 200000, 'small': 11001, 'short': 2501, 'tiny': 32 * 3 + 1 + 32 * 2}
    if dataset is '/celeba-low':
        path = Path(data_root + dataset)
        """ celeba a transforms """
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        data = tv.datasets.ImageFolder(str(path), transform=transform)
    elif dataset is 'square':
        data = SquareDataset(size=200000, transform=transforms.ToTensor())
    elif dataset is 'pong':
        data = AtariDataset('Pong-v0', size[run_type], pong_prepro)
    else:
        raise ConfigException('pick a dataset')

    if run_type == 'full':
        train = torch.utils.data.Subset(data, range(190000))
        test = torch.utils.data.Subset(data, range(190001, len(data)))
    elif run_type == 'small':
        train = torch.utils.data.Subset(data, range(10000))
        test = torch.utils.data.Subset(data, range(10001, 11001))
    elif run_type == 'short':
        train = torch.utils.data.Subset(data, range(2000))
        test = torch.utils.data.Subset(data, range(2001, 2501))
    elif run_type == 'tiny':
        train = torch.utils.data.Subset(data, range(32 * 3))
        test = torch.utils.data.Subset(data, range(32 * 3 + 1, 32 * 3 + 1 + 32 * 2))
    else:
        raise ConfigException('pick a run type')

    return train, test
