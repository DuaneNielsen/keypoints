import collections
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.utils.data
import torchvision as tv
from torchvision import transforms
import gym
import skimage.measure
from tqdm import tqdm

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
    s = skimage.measure.block_reduce(s, (4, 4), np.max)
    s = cv2.resize(s, dsize=(32, 32), interpolation=cv2.INTER_AREA)
    return s


def pong_color_prepro(s):
    s = s[:, 25:, :]
    s = cv2.resize(s, dsize=(32, 32), interpolation=cv2.INTER_LINEAR)
    return s


def if_done(done, r):
    return done


def if_done_or_nonzero_reward(done, r):
    return done or r != 0


class AtariDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, env, size, prepro, transforms=None, end_trajectory=if_done, skip_first=False):
        super().__init__()
        self.loadbar = tqdm(total=size, desc='LOADING TRAJECTORIES FROM SIMULATOR')
        self.prepro = prepro
        self.end_trajectory = end_trajectory
        self.env = gym.make(env)
        self.env.reset()
        self.trajectories = []
        self.index = []
        self.skip_first = skip_first
        self._fill(size)
        self.transforms = transforms


    def _trajectory(self, reset):
        frames = []
        if reset:
            self.env.reset()
        done = False
        r = 0

        while not self.end_trajectory(done, r):
            s, r, done, info = self.env.step(self.env.action_space.sample())
            ps = self.prepro(s)
            frames.append(ps)

        return np.stack(frames, axis=0), done

    def __len__(self):
        return len(self.index)

    def _fill(self, samples):
        traj_id = 0
        done = False
        first = True

        while len(self) < samples:
            t, done = self._trajectory(reset=done)
            if not first:
                self.trajectories.append(t)
                first = done
                for i in range(t.shape[0] - 1):
                    self.index.append((traj_id, i))
                traj_id += 1
                self.loadbar.update(t.shape[0])
            else:
                # skip the first trajectory cos it's suspect
                first = False

    def __getitem__(self, item):
        trajectory, i = self.index[item]
        t, t1 = self.trajectories[trajectory][i], self.trajectories[trajectory][i + 1]
        if len(t.shape) == 2:
            # add a channel dimension if grayscale
            # expected shape is H, W, C
            t, t1 = t.reshape(*t.shape, 1), t1.reshape(*t1.shape, 1)
        if self.transforms is not None:
            t, t1 = self.transforms(t), self.transforms(t1)
        return t, t1


D_CELEBA = 'celeba'
D_SQUARE = 'square'
D_PONG = 'pong'

SIZE_FULL = 'full'
SIZE_SMALL = 'small'
SIZE_SHORT = 'short'
SIZE_TINY = 'tiny'


def get_dataset(data_root, dataset, run_type):
    size = {'full': 200000, 'small': 11001, 'short': 2501, 'tiny': 32 * 3 + 1 + 32 * 2}
    if dataset == 'celeba':
        path = Path(data_root + '/celeba-low')
        """ celeba a transforms """
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        data = tv.datasets.ImageFolder(str(path), transform=transform)
    elif dataset == 'square':
        data = SquareDataset(size=200000, transform=transforms.ToTensor())
    elif dataset == 'pong':
        data = AtariDataset('Pong-v0', size[run_type], pong_prepro, transforms=transforms.ToTensor())
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
