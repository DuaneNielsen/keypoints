import models.functional as MF
from models import knn, losses
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from tps import tps_grid, tps_random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from utils import plot_keypoints_on_image, UniImageViewer
import torchvision.transforms as tvt
from tests.common import bad_monkey
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from matplotlib.gridspec import GridSpec

heatmap_batch = torch.tensor([
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


def heatmap(value=1.0, batch=1, channels=1):
    return torch.tensor([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, value, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]).expand(batch, channels, 5, 5).float()




def test_plot():
    z = np.array([[x ** 2 + y ** 2 for x in range(20)] for y in range(20)])
    x, y = np.meshgrid(range(z.shape[0]), range(z.shape[1]))

    # show height map in 3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z)
    plt.title('z as 3d height map')
    plt.show()


def test_perceptual_loss():
    criterion = losses.PerceptualLoss()
    x = torch.rand(1, 3, 256, 256)
    target = torch.rand(1, 3, 256, 256)
    loss = criterion(x, target)
    print(f'loss: {loss.item()}')
    # assert loss.item() == F.mse_loss(x, target).item()


""" SPACIAL BOTTLENECK """


def test_spacial_softmax():
    heatmap = torch.tensor([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 100, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]).expand(1, 1, 5, 5).float()

    ss = knn.SpatialSoftmax()

    k = ss(heatmap)

    print(k)


def test_spacial_softmax():

    batch = 5
    channels = 10

    ss = knn.SpatialSoftmax()
    k = ss(heatmap(batch=batch, channels=channels))

    print(k)
    assert k.size(0) == batch
    assert k.size(1) == channels
    assert k.size(2) == 2


def test_gaussian_like():
    heatmap = torch.tensor([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 5, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]).expand(1, 1, 5, 5).float()

    ss = knn.SpatialSoftmax()

    k = ss(heatmap)

    hm = MF.gaussian_like_function(k, 5, 5)

    print('')
    print(hm)


def test_gaussian_like_batch():
    ss = knn.SpatialSoftmax()

    k = ss(heatmap_batch)

    hm = MF.gaussian_like_function(k, 5, 5)

    print('')
    print(hm)


def test_spacial_softmax_grads():
    heatmap = torch.rand(1, 1, 5, 5, requires_grad=True)
    h = heatmap.neg()
    ss = knn.SpatialSoftmax()
    k = ss(h)
    loss = torch.sum(torch.cat(k, dim=1))
    loss.backward()
    print(heatmap)
    print(heatmap.grad)


def test_gaussian_function_grads():
    x, y = torch.rand(1, 5, requires_grad=True), torch.rand(1, 5, requires_grad=True)
    kp = x.neg(), y.neg()
    ss = MF.gaussian_like_function(kp, 5, 5)
    loss = torch.sum(ss)
    loss.backward()
    print(ss)
    print(x, y)
    print(x.grad, y.grad)


def test_plot_gaussian_function():
    mu = 0.5
    sigma = 0.4
    kp = torch.randn(1, 1, 2, requires_grad=True) * sigma + mu
    z = MF.gaussian_like_function(kp, 14, 14, sigma=0.1).squeeze().detach().numpy()

    plot_heightmap3d(z)
    plot_heatmap2d(z)


def test_align_kp_with_gaussian():
    hm = heatmap()

    # image = TVF.to_pil_image(heatmap[0])

    ss = knn.SpatialSoftmax()

    kp = ss(hm)

    z = MF.gaussian_like_function(kp, 5, 5, sigma=0.1).squeeze().detach().numpy()

    img = plot_keypoints_on_image(kp[0], hm[0])
    plt.imshow(img)
    plot_heatmap2d(z)


def plot_heatmap2d(z):
    # show height map in 2d
    plt.figure()
    plt.title('z as 2d heat map')
    p = plt.imshow(z)
    plt.colorbar(p)
    plt.show()


def plot_heightmap3d(z, k=None):
    # show height map in 3d
    coordinates = np.meshgrid(range(z.shape[0]), range(z.shape[1]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(*coordinates, z)

    if k is not None:
        plt.title(f'z as 3d height map {k[0]} {k[1]}')
    else:
        plt.title('z as 3d height map')
    plt.show()


def plot_single_channel(tensor):
    plt.imshow(tensor.numpy(), cmap='gray', vmin=0, vmax=1)
    plt.show()

def plot_marginal(tensor):
    plt.hist(tensor.numpy())
    plt.show()

def plot_joint(image, x_marginal, y_marginal):

    fig = plt.figure()

    gs = GridSpec(4, 4)

    ax_joint = fig.add_subplot(gs[1:4, 0:3])
    ax_marg_top = fig.add_subplot(gs[0, 0:3])
    ax_marg_side = fig.add_subplot(gs[1:4, 3])

    ax_marg_side.set_autoscaley_on(False)
    #ax_marg_side.set_ylim([0, 32])

    ax_marg_top.set_autoscaley_on(False)
    #ax_marg_top.set_ylim([0, 32])

    ax_joint.imshow(image, cmap='gray', vmin=0, vmax=1)
    ax_marg_top.hist(x_marginal)
    ax_marg_side.hist(y_marginal, orientation="horizontal")

    # Turn off tick labels on marginals
    #plt.setp(ax_marg_y.get_xticklabels(), visible=False)
    #plt.setp(ax_marg_x.get_yticklabels(), visible=False)


    # Set labels on joint
    ax_joint.set_xlabel('Joint x label')
    ax_joint.set_ylabel('Joint y label')

    # Set labels on marginals
    ax_marg_side.set_xlabel('Marginal side')
    ax_marg_top.set_ylabel('Marginal top')
    plt.show()

def test_marginals():
    img = np.random.random((4, 4))
    x_marginal = np.mean(img, axis=0)
    y_marginal = np.mean(img, axis=1)
    plot_joint(img, x_marginal, y_marginal)


def test_co_ords():
    height, width = 16, 16
    hm = torch.zeros(1, 1, height, width)
    hm[0, 0, 0, 15] = 1.0
    k, p = MF.spacial_softmax(hm, probs=True)
    g = MF.gaussian_like_function(k, height, width)
    #plot_heightmap3d(hm[0, 0].detach().numpy())
    #plot_heightmap3d(g[0, 0].detach().numpy(), k[0, 0])
    #plot_single_channel(hm[0, 0])
    #plot_single_channel(g[0, 0])
    #plot_joint(hm[0, 0], p[1][0], p[0][0])
    plot_marginal(p[0][0])



def test_bottlneck_grads():
    heatmap = torch.rand(1, 1, 5, 5, requires_grad=True)
    h = heatmap.neg()
    ss = knn.SpatialSoftmax()
    kp = ss(h)
    ss = MF.gaussian_like_function(kp, 5, 5)
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


"""  DISPLAY THE KEYPOINTS """


def test_plt_keypoints():
    num_monkeys = 1
    bm = bad_monkey(num_monkeys)
    height, width = bm.size(2), bm.size(3)
    heatmap = torch.rand(num_monkeys, 10, height, width, requires_grad=False)
    ss = knn.SpatialSoftmax()
    k = ss(heatmap)
    image = plot_keypoints_on_image(k[0], bm[0], radius=3, thickness=3)
    plt.imshow(image)
    plt.show()


""" THIN PLATE SPLINES"""


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
            set.append(tps_random(x, num_control_points=20, var=1 / i))

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
