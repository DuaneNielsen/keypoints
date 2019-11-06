import models
import torch
import torch.nn.functional as F
import colorama
import cv2
from utils import UniImageViewer
from pathlib import Path
from os import getcwd
from torchvision.transforms import ToTensor
from tps import tps_grid, tps_random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def test_plot():
    z = np.array([[x**2 + y**2 for x in range(20)] for y in range(20)])
    x, y = np.meshgrid(range(z.shape[0]), range(z.shape[1]))

    # show height map in 3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z)
    plt.title('z as 3d height map')
    plt.show()


def test_perceptual_loss():
    criterion = models.PerceptualLoss()
    x = torch.rand(1, 3, 256, 256)
    target = torch.rand(1, 3, 256, 256)
    loss = criterion(x, target)
    print(f'loss: {loss.item()}')
    # assert loss.item() == F.mse_loss(x, target).item()


def test_spacial_softmax():
    heatmap = torch.tensor([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 100, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]).expand(1, 1, 5, 5).float()

    ss = models.SpatialSoftmax(5, 5)

    k = ss(heatmap)

    print(k)


def test_gaussian_like():
    heatmap = torch.tensor([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 5, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]).expand(1, 1, 5, 5).float()

    ss = models.SpatialSoftmax(5, 5)

    k = ss(heatmap)

    hm = models.gaussian_like_function(k, 5, 5)

    print('')
    print(hm)


def test_gaussian_like_batch():
    heatmap = torch.tensor([
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 5, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5],
        ],

    ]).expand(2, 2, 5, 5).float()

    ss = models.SpatialSoftmax(5, 5)

    k = ss(heatmap)

    hm = models.gaussian_like_function(k, 5, 5)

    print('')
    print(hm)

def test_spacial_softmax_grads():
    heatmap = torch.rand(1, 1, 5, 5, requires_grad=True)
    h = heatmap.neg()
    ss = models.SpatialSoftmax(5, 5)
    k = ss(h)
    loss = torch.sum(torch.cat(k, dim=1))
    loss.backward()
    print(heatmap)
    print(heatmap.grad)


def test_gaussian_function_grads():
    x, y = torch.rand(1, 5, requires_grad=True), torch.rand(1, 5, requires_grad=True)
    kp = x.neg(), y.neg()
    ss = models.gaussian_like_function(kp, 5, 5)
    loss = torch.sum(ss)
    loss.backward()
    print(ss)
    print(x, y)
    print(x.grad, y.grad)

def test_plot_gaussian_function():
    mu = 0.5
    sigma = 0.4
    kp = torch.randn(1, 1, requires_grad=True) * sigma + mu, torch.randn(1, 1, requires_grad=True) * sigma + mu
    #kp = torch.zeros(1, 1, requires_grad=True), torch.zeros(1, 1, requires_grad=True)
    z = models.gaussian_like_function(kp, 14, 14, sigma=0.2).squeeze().detach().numpy()
    coordinates = np.meshgrid(range(z.shape[0]), range(z.shape[1]))

    # show height map in 3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(*coordinates, z)
    plt.title('z as 3d height map')
    plt.show()

    # show height map in 2d
    plt.figure()
    plt.title('z as 2d heat map')
    p = plt.imshow(z)
    plt.colorbar(p)
    plt.show()

def test_bottlneck_grads():
    heatmap = torch.rand(1, 1, 5, 5, requires_grad=True)
    h = heatmap.neg()
    ss = models.SpatialSoftmax(5, 5)
    kp = ss(h)
    ss = models.gaussian_like_function(kp, 5, 5)
    loss = torch.sum(ss)
    loss.backward()
    print(heatmap.grad)
    print(kp[0].grad, kp[1].grad)

def test_intermediate_grads():
    heatmap = torch.rand(1, 5, requires_grad=True)
    l1 = torch.nn.Linear(5, 5)
    l2 = torch.nn.Linear(5, 5)
    activation = l1(heatmap)
    out = l2(activation)
    loss = torch.sum(out)
    loss.backward()
    print(activation.grad)
    print(heatmap.grad)


def bad_monkey(copies=1):
    p = Path('/home/duane/tv-data/bad_monkey.jpg')
    assert p.exists()
    x = cv2.imread(str(p))
    x = ToTensor()(x).expand(copies, -1, -1, -1)
    return x


def test_flowfield():
    u = UniImageViewer()
    x = bad_monkey()

    theta = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ]).expand(1, -1, -1)

    grid = F.affine_grid(theta, x.shape)

    out = F.grid_sample(x, grid)

    u.render(out[0], block=True)


def test_tps():
    u = UniImageViewer()
    x = bad_monkey()

    theta = torch.tensor([[
        [0.0, 0.0],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0.0, 0.0]]])

    c = torch.tensor([
        [0., 0],
        [1., 0],
        [1., 1],
        [0, 1],
    ]).unsqueeze(0)

    grid = tps_grid(theta, c, x.shape)

    out = F.grid_sample(x, grid)

    u.render(out[0], block=True)

def test_tps_random():

    images = []
    u = UniImageViewer(screen_resolution=(2400, 1200))
    x = bad_monkey()

    for i in range(5, 10):

        set = []

        for _ in range(8):
            set.append(tps_random(x, num_control_points=20, var=1/i))

        st = torch.cat(set, dim=2)
        images.append(st)

    img = torch.cat(images, dim=3)

    u.render(img, block=True)


def test_dual_tps_random():
    images = []
    u = UniImageViewer(screen_resolution=(2400, 1200))
    x = bad_monkey()

    for i in range(15, 20):

        set = []

        for _ in range(4):
            img1 = tps_random(x, num_control_points=20, var=1 / i)
            img2 = tps_random(img1, num_control_points=20, var=1 / i)

            set.append(torch.cat((img1, img2), dim=2))

        st = torch.cat(set, dim=2)
        images.append(st)

    img = torch.cat(images, dim=3)

    u.render(img, block=True)


def test_dual_tps_random_batched():
    u = UniImageViewer(screen_resolution=(2400, 1200))
    x = bad_monkey(2)

    img1 = tps_random(x, num_control_points=20, var=1 / 20)
    img2 = tps_random(img1, num_control_points=20, var=1 / 20)

    img = torch.cat((torch.cat(img1.unbind(0), dim=2), torch.cat(img2.unbind(0), dim=2)), dim=1)

    u.render(img, block=True)
