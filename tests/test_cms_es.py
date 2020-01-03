from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import cma_es
from matplotlib.patches import Ellipse

def objective(x, y):
    return 1 / ((x * 2.0 + 0.3) ** 2 + (y * 2.0 + 0.4) ** 2).sqrt()


def test_plot_objective():
    x_, y_, = torch.linspace(-1, 1, 100), torch.linspace(-1, 1, 100)
    x, y = torch.meshgrid([x_, y_])
    z = objective(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x.cpu().numpy(), y.cpu().numpy(), z.cpu().numpy())
    plt.show()
    plt.imshow(z, cmap='hot')
    plt.show()


def plot_heatmap(samples, b, d):
    axis_scale = 1.2
    x_, y_, = torch.linspace(-1, 1, 100), torch.linspace(-1, 1, 100)
    x, y = torch.meshgrid([x_, y_])
    z = objective(x, y)
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    ax2.contour(x, y, z, cmap='hot')
    ax2.scatter(samples[:, 0], samples[:, 1])
    covar = Ellipse(xy=(0.0, 0.0), width=1.0, height=1.0, alpha=0.2)
    ax2.add_artist(covar)
    ax2.set_xlim(-axis_scale, axis_scale)
    ax2.set_ylim(-axis_scale, axis_scale)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def test_map():
    features = 2
    mean = torch.zeros(features)
    b = torch.eye(features)
    d = torch.eye(features)

    s, z = cma_es.sample(5, 0.3, mean, b, d)
    plot_heatmap(s, b, d)