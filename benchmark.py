import collections
import numpy as np
import cv2
import torch.utils.data

Pos = collections.namedtuple('Pos', 'x, y')


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
