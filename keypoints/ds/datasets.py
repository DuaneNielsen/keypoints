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

import cma_es
from keypoints import models as MF

Pos = collections.namedtuple('Pos', 'x, y')


class ConfigException(Exception):
    pass


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
    """ generates a white square in the center of the image """

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


def nop(s):
    return s


def pong_prepro(s):
    s = cv2.cvtColor(s, cv2.COLOR_RGB2GRAY)
    s = s[34:168, :]
    # s = skimage.measure.block_reduce(s, (4, 4), np.max)
    s = cv2.resize(s, dsize=(32, 32), interpolation=cv2.INTER_AREA)
    return s


def pong_color_prepro(s):
    s = s[34:168, :]
    s = skimage.measure.block_reduce(s, (4, 4, 1), np.max)
    s = cv2.resize(s, dsize=(32, 32), interpolation=cv2.INTER_AREA)
    return s


def pacman_color_prepro(s):
    s = s[:171, :]
    s = cv2.resize(s, dsize=(128, 128), interpolation=cv2.INTER_AREA)
    return s


def if_done(done, r):
    return done


def if_done_or_nonzero_reward(done, r):
    return done or r != 0


class AtariDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, env, size, prepro, max_frame_skip=20, min_frame_skip=5, transforms=None,
                 end_trajectory=if_done, skip_first=False):
        """

        :param env: gym env string
        :param size: total length of dataset to generate
        :param prepro: preprocessing function, numpy RGB input, numpy RGB output
        :param max_frame_skip: when generating paired images, skip multiple frames up to max
        :param min_frame_skip: skip multiple frames up to min
        :param transforms: pytorch transform pipeline, takes in numpy RGB and returns tensor
        :param end_trajectory: function that takes in done and reward for timestep and returns done flag
        :param skip_first: skip the first episode
        """
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


class MapperDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, height, width, keypoints, length):
        super().__init__()
        self.height = height
        self.width = width
        self.keypoints = keypoints
        self.length = length

    def __getitem__(self, item):
        kp = torch.rand(1, self.keypoints, 2)
        pointmap = MF.point_map(kp, self.height, self.width)
        mask = MF.gaussian_like_function(kp, self.height, self.width)
        mask, _ = torch.max(mask, dim=1, keepdim=True)
        return pointmap.squeeze(0), mask.squeeze(0)

    def __len__(self):
        return self.length


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
    return [torch.utils.data.Subset(dataset, indices[offset - length:offset]) for offset, length in
            zip(_accumulate(lengths), lengths)]


def split(data, train_len, test_len):
    total_len = train_len + test_len
    train = torch.utils.data.Subset(data, range(0, train_len))
    test = torch.utils.data.Subset(data, range(train_len, total_len))
    return train, test


class DataPack(object):
    def __init__(self):
        self.name = None
        self.transforms = None

    def make(self, train_len, test_len, **kwargs):
        pass


class ImageDataPack(DataPack):
    def __init__(self, name, subdir, transforms):
        super().__init__()
        self.name = name
        self.transforms = transforms
        self.subdir = subdir

    def make(self, train_len, test_len, data_root='data', **kwargs):
        """
        Returns a test and training dataset
        :param train_len: images in training set
        :param test_len: images in test set
        :param data_root: the the root directory for project datasets
        :return:
        """

        data = tv.datasets.ImageFolder(str(Path(data_root) / Path(self.subdir)), transform=self.transforms, **kwargs)
        return split(data, train_len, test_len)


class AtariDataPack(DataPack):
    def __init__(self, name, env, prepro, transforms, action_map):
        super().__init__()
        self.name = name
        self.env = env
        self.prepro = prepro
        self.transforms = transforms
        self.action_map = action_map
        self.shape = (3, 210, 160)

    def make(self, train_len, test_len, *args, **kwargs):
        total_len = train_len + test_len
        data = AtariDataset(self.env, total_len, self.prepro,
                            end_trajectory=if_done,
                            transforms=self.transforms)
        return random_split(data, (train_len, test_len))


class SquareDataPack(DataPack):
    def __init__(self):
        super().__init__()
        self.name = 'square'
        self.transforms = transforms.ToTensor()

    def make(self, train_len, test_len, *args, **kwargs):
        total_len = train_len + test_len
        data = SquareDataset(total_len, transform=self.transforms)
        return split(data, train_len, test_len)


color_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

grey_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

celeba_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


class ActionMap(object):
    def __init__(self, size=None, f=None, map=None):
        super().__init__()
        self.size = size if size else len(map)
        self.map = map
        # pickle hates lambdas, so we going to...
        if f:
            self.f = f
        elif map is not None:
            self.f = self._mapf
        else:
            self.f = self._nop

    def _nop(self, a):
        return a

    def _mapf(self, a):
        return self.map[a]

    def __call__(self, a):
        return self.f(a)


def pong_action_map(a):
    return a + 2


nop_left_right = [0, 2, 3]  # 'NOOP', 'LEFT', 'RIGHT'


def pacman_action_map(a):
    return a + 1


def box2d_discrete(a):
    return a.item()


datasets = {
    'celeba': ImageDataPack('celeba', 'celeba-low', celeba_transform),
    'square': SquareDataPack(),
    'pong': AtariDataPack('pong', 'Pong-v0', pong_prepro, grey_transform, ActionMap(2, pong_action_map)),
    'pong_color': AtariDataPack('pong_color', 'Pong-v0', pong_color_prepro, color_transform, ActionMap(2, pong_action_map)),
    'pong_color_3_act': AtariDataPack('pong_color', 'Pong-v0', pong_color_prepro, color_transform, ActionMap(map=nop_left_right)),
    'pacman': AtariDataPack('pacman', 'MsPacman-v0', pacman_color_prepro, color_transform, ActionMap(4, pacman_action_map)),
    'cartpole': AtariDataPack('cartpole', 'CartPole-v0', nop, torch.from_numpy, ActionMap(2, box2d_discrete))
}