from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import cma_es
from matplotlib.patches import Ellipse
from math import cos, sin, acos, degrees
import time

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


def plot_heatmap(title, count, mean, b, d, samples=None, g=None):
    axis_scale = 1.2
    x_, y_, = torch.linspace(-1, 1, 100), torch.linspace(-1, 1, 100)
    x, y = torch.meshgrid([x_, y_])
    z = objective(x, y)
    fig = plt.figure()
    fig.suptitle(f'{title} {count}', fontsize=16)
    ax2 = fig.add_subplot(111)
    ax2.contour(x, y, z, cmap='hot')
    if samples is not None:
        ax2.scatter(samples[:, 0], samples[:, 1], color='blue')
    if g is not None:
        ax2.scatter(g[:, 0], g[:, 1], color='red')

    trans = torch.tensor([
        [1.0, 0, mean[0]],
        [0, 1.0, mean[1]],
        [0, 0, 1.0]
    ])

    x_unit = torch.tensor([
            [0.0, 1.0],
            [0.0, 0],
    ])

    y_unit = torch.tensor([
            [0, 0.0],
            [0, 1.0],
    ])

    units = torch.stack((x_unit, y_unit))

    theta = acos(b[0, 0])

    unit = b.matmul(d.matmul(units))
    unit = torch.cat((unit, torch.ones(2, 1, 2)), dim=1)
    unit = trans.matmul(unit)

    ax2.scatter(mean[0], mean[1], color='green')

    xunit_x, xunit_y = unit[0, 0], unit[0, 1]
    yunit_x, yunit_y = unit[1, 0], unit[1, 1]

    ax2.plot(xunit_x, xunit_y)
    ax2.plot(yunit_x, yunit_y)

    covar = Ellipse(xy=(mean[0], mean[1]), width=d[0, 0] * 2, height=d[1, 1] * 2, angle=-degrees(theta), alpha=0.2)
    ax2.add_artist(covar)

    max_g_x = g[:, 0].abs().max() if g is not None else 0
    max_g_y = g[:, 1].abs().max() if g is not None else 0

    xscale = max(xunit_x.abs().max().item(), yunit_x.abs().max().item(), 0.3, max_g_x) * 1.1
    yscale = max(xunit_y.abs().max().item(), yunit_y.abs().max().item(), 0.4, max_g_y) * 1.1

    ax2.set_xlim(-xscale + mean[0], xscale + mean[0])
    ax2.set_ylim(-yscale + mean[1], yscale + mean[1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def test_sampler():
    features = 2
    samples = 16
    step_size = 1.0

    mean = torch.zeros(features)
    b = torch.eye(features)
    d = torch.eye(features)

    for counteval in range(4):
        s, z = cma_es.sample(samples, step_size, mean, b, d)
        f = objective(s[:, 0], s[:, 1])
        g = [{'sample': s[i], 'fitness':f.item()} for i, f in enumerate(f)]
        g = sorted(g, key=lambda x: x['fitness'], reverse=True)
        g = g[0:samples//4]
        g = torch.stack([g['sample'] for g in g])
        plot_heatmap('sample ', counteval, mean, b, d, samples=s, g=g)
        mean_prev = mean.clone()
        g_raw = g.clone()
        mean = g.mean(0)
        g = g - mean_prev
        c = g.T.matmul(g) * 4 / samples
        d, b = torch.symeig(c, eigenvectors=True)
        d = d.sqrt().diag_embed()
        plot_heatmap('select', counteval, mean, b, d, g=g_raw)
        time.sleep(0.5)


