import collections
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.utils.data
from torch import randperm
from torch._utils import _accumulate
import torch.random
import torchvision as tv
from torchvision import transforms
import gym
import skimage.measure
from tqdm import tqdm
from random import randint

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
    s = s[34:168, :]
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
    def __init__(self, env, size, prepro, max_frame_skip=20, min_frame_skip=5, transforms=None,
                 end_trajectory=if_done, skip_first=False):
        super().__init__()
        self.loadbar = tqdm(total=size, desc='Loading trajectories from sim')
        self.prepro = prepro
        self.end_trajectory = end_trajectory
        self.env = gym.make(env)
        self.env.reset()
        self.trajectories = []
        self.index = []
        self.skip_first = skip_first
        self.max_frame_skip = max_frame_skip
        self.min_frame_skip = min_frame_skip
        self.transforms = transforms

        self._fill(size)

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
                for i in range(t.shape[0] - self.max_frame_skip):
                    self.index.append((traj_id, i))
                traj_id += 1
                self.loadbar.update(t.shape[0])
            else:
                # skip the first trajectory cos it's suspect
                first = False

    def __getitem__(self, item):
        trajectory, i = self.index[item]
        frame_skip = randint(self.min_frame_skip, self.max_frame_skip)
        t, t1 = self.trajectories[trajectory][i], self.trajectories[trajectory][i + frame_skip]
        if len(t.shape) == 2:
            # add a channel dimension if grayscale
            # expected shape is H, W, C
            t, t1 = t.reshape(*t.shape, 1), t1.reshape(*t1.shape, 1)
        if self.transforms is not None:
            t, t1 = self.transforms(t), self.transforms(t1)
        return t, t1


def random_split(dataset, lengths):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) > len(dataset):
        raise ValueError("Sum of input lengths is greater than the length of the input dataset!")

    indices = randperm(sum(lengths)).tolist()
    return [torch.utils.data.Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]



D_CELEBA = 'celeba'
D_SQUARE = 'square'
D_PONG = 'pong'


def get_dataset(data_root, dataset, train_len, test_len, randomize=False):

    total_len = train_len + test_len

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
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        data = AtariDataset('Pong-v0', total_len, pong_prepro,
                            end_trajectory=if_done_or_nonzero_reward,
                            transforms=transform)
    else:
        raise ConfigException('pick a dataset')

    if total_len > len(data):
        raise ConfigException(f'total length in config is {total_len} but dataset has only {len(data)} entries')

    if randomize:
        train, test = random_split(data, (train_len, test_len))
    else:
        train = torch.utils.data.Subset(data, range(0, train_len))
        test = torch.utils.data.Subset(data, range(train_len, total_len))

    return train, test
