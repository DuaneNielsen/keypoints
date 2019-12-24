import utils as u
import tests.common as com
import torch
from torchvision.utils import make_grid

def test_display_keypoints():
    x = com.bad_monkey()
    d = u.ResultsLogger('model_name', 'run_id')
    k = com.keypoints()
    d.display(x, blocking=True)


def test_color_map():
    cm = u.color_map()
    print(cm)


def test_resize():
    x = com.bad_monkey()
    x = u.resize2D(x[0], (512, 512))
    d = u.ResultsLogger('model_name', 'run_id')
    d.display(x, blocking=True)


def test_panel():
    t = torch.linspace(1, 9, 9).reshape(1, 3, 3)
    x = torch.stack((t, t* 3, t * 5, t * 7))
    index = torch.arange(min(x.size(0), 2))
    panel = make_grid(x, nrow=3, padding=0)
    #panel = u.panel_tensor(x, 1, 2)

    print(panel)

import numpy as np
import torchvision
from matplotlib import pyplot as plt

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show()

def test_make_grid():
    w = torch.randn(8,1,640,640)
    grid = torchvision.utils.make_grid(w, nrow=10, padding=100)
    show(grid)