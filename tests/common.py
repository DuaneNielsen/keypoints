from pathlib import Path
import cv2
from torchvision.transforms import ToTensor

from keypoints.models import knn
import torch

""" DATA  """


def bad_monkey(copies=1):
    p = Path('600px-Bad_Monkey_Gaming.png')
    print(Path.cwd())
    assert p.exists()
    x = cv2.imread(str(p))
    x = ToTensor()(x).expand(copies, -1, -1, -1)
    return x


def keypoints(num=10):
    bm = bad_monkey()
    height, width = bm.size(2), bm.size(3)
    heatmap = torch.rand(1, num, height, width, requires_grad=False)
    h = heatmap.neg()
    ss = knn.SpatialSoftmax()
    x, y = ss(h)
    return x, y