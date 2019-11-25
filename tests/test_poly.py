import numpy as np
import cv2
import matplotlib.pyplot as plt
import benchmark as b
from benchmark import Pos, SquareDataset
from torch.utils.data import DataLoader


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
