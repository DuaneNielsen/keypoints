from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import cma_es
from matplotlib.patches import Ellipse
from math import cos, sin, acos, degrees, log, floor, sqrt
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

    ax2.scatter(mean[0], mean[1], color='yellow')

    xunit_x, xunit_y = unit[0, 0], unit[0, 1]
    yunit_x, yunit_y = unit[1, 0], unit[1, 1]

    ax2.plot(xunit_x, xunit_y)
    ax2.plot(yunit_x, yunit_y)

    covar = Ellipse(xy=(mean[0], mean[1]), width=d[0, 0] * 2, height=d[1, 1] * 2, angle=-degrees(theta), alpha=0.2)
    ax2.add_artist(covar)

    max_g_x = g[:, 0].abs().max() if g is not None else 0
    max_g_y = g[:, 1].abs().max() if g is not None else 0
    max_s_x = samples[:, 0].abs().max() if samples is not None else 0
    max_s_y = samples[:, 1].abs().max() if samples is not None else 0

    xscale = max(xunit_x.abs().max().item(), yunit_x.abs().max().item(), 0.3, max_g_x, max_s_x) * 1.1
    yscale = max(xunit_y.abs().max().item(), yunit_y.abs().max().item(), 0.4, max_g_y, max_s_y) * 1.1

    ax2.set_xlim(-xscale + mean[0], xscale + mean[0])
    ax2.set_ylim(-yscale + mean[1], yscale + mean[1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def test_rank_mu_update():
    features = 2

    step_size = 1.0
    epochs = 1e3 * features ** 2

    # selection settings
    samples = 4 + floor(3 * log(features))
    mu = samples / 2
    weights = log(mu + 0.5) + torch.linspace(start=1, end=mu, steps=floor(mu)).log()
    weights = torch.flip(weights, dims=(0,)) / weights.sum()
    mu = floor(mu)
    mueff = weights.sum() ** 2 / (weights ** 2).sum()

    # adaptation settings
    cmu = mueff / features ** 2
    print(cmu)
    plt.title('weights')
    plt.plot(weights)
    plt.show()

    mean = torch.zeros(features)
    b = torch.eye(features)
    d = torch.eye(features)
    c = torch.matmul(b.matmul(d), b.matmul(d).T)

    for counteval in range(4):

        # sample parameters
        s, z = cma_es.sample(samples, step_size, mean, b, d)

        # rank by fitness
        f = objective(s[:, 0], s[:, 1])
        g = [{'sample': s[i], 'z': z[i], 'fitness':f.item()} for i, f in enumerate(f)]
        g = sorted(g, key=lambda x: x['fitness'], reverse=True)
        g = g[0:mu]
        z = torch.stack([g['z'] for g in g])
        g = torch.stack([g['sample'] for g in g])
        plot_heatmap('sample ', counteval, mean, b, d, samples=s, g=g)

        c_prev = c.clone()
        g_raw = g.clone()

        mean = (g * weights.unsqueeze(1)).sum(0)
        zmean = (z * weights.unsqueeze(1)).sum(0)

        # estimate weighted covariance in z-space
        t = b.matmul(d).matmul(z.t())
        c = torch.matmul(t, weights.diag_embed())
        c = c.matmul(t.t())

        c = (1.0 - cmu) * c_prev + cmu * c
        d, b = torch.symeig(c, eigenvectors=True)
        d = d.sqrt().diag_embed()
        plot_heatmap('select', counteval, mean, b, d, g=g_raw)
        time.sleep(0.5)


def test_rank_one_update():
    features = 2

    step_size = 1.0
    epochs = 1e3 * features ** 2

    # selection settings
    samples = 4 + floor(3 * log(features))
    mu = samples / 2
    weights = log(mu + 0.5) + torch.linspace(start=1, end=mu, steps=floor(mu)).log()
    weights = torch.flip(weights, dims=(0,)) / weights.sum()
    mu = floor(mu)
    mueff = (weights.sum() ** 2 / (weights ** 2).sum()).item()

    '''
    cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N);
    cs = (mueff + 2) / (N + mueff + 5);
    c1 = 2 / ((N + 1.3)ˆ2+mueff);
    cmu = 2 * (mueff - 2 + 1 / mueff) / ((N + 2)ˆ2+2 * mueff / 2);
    damps = 1 + 2 * max(0, sqrt((mueff - 1) / (N + 1)) - 1) + cs;
    '''

    # adaptation settings
    #cmu = mueff / features ** 2
    cc = (4 + mueff/features) / (features+4 + 2 * mueff/features)
    cs = (mueff + 2) / (features + mueff + 5)
    # c1 = 2 / ((features + 1.3) ** 2 + mueff)
    c1 = 0.5
    cmu = 2 * (mueff - 2 + 1 / mueff) / ((features + 2)**2 + 2 * mueff / 2)
    damps = 1 + 2 * max(0.0, sqrt((mueff - 1.0) / (features + 1)) -1 ) + cs

    print(f'cc : {cc}, cs: {cs}, c1: {c1}, cmu: {cmu}, damps: {damps}')

    plt.title('weights')
    plt.plot(weights)
    plt.show()

    mean = torch.zeros(features)
    b = torch.eye(features)
    d = torch.eye(features)
    c = torch.matmul(b.matmul(d), b.matmul(d).T)

    pc = torch.zeros(features)

    for counteval in range(8):

        # sample parameters
        s, z = cma_es.sample(samples, step_size, mean, b, d)

        # rank by fitness
        f = objective(s[:, 0], s[:, 1])
        g = [{'sample': s[i], 'z': z[i], 'fitness':f.item()} for i, f in enumerate(f)]
        g = sorted(g, key=lambda x: x['fitness'], reverse=True)
        g = g[0:mu]
        z = torch.stack([g['z'] for g in g])
        g = torch.stack([g['sample'] for g in g])
        plot_heatmap('sample ', counteval, mean, b, d, samples=s, g=g)

        mean_prev = mean.clone()
        c_prev = c.clone()
        g_raw = g.clone()

        mean = (g * weights.unsqueeze(1)).sum(0)
        zmean = (z * weights.unsqueeze(1)).sum(0)

        # a mind bending way to write a exponential smoothed moving average for the variance
        # zmean does not contain step size or mean, so allows us to add together
        # updates of different step sizes
        pc = (1 - cc) * pc + cc * b.matmul(d).matmul(zmean)
        cov_pc = pc.unsqueeze(1).matmul(pc.unsqueeze(1).t())

        # update covariance from smoothed mean in zspace
        c = (1 - c1) * c + c1 * cov_pc


        # estimate weighted covariance in z-space
        # t = b.matmul(d).matmul(z.t())
        # c = torch.matmul(t, weights.diag_embed())
        # c = c.matmul(t.t())
        # c = (1.0 - cmu) * c_prev + cmu * c


        d, b = torch.symeig(c, eigenvectors=True)
        d = d.sqrt().diag_embed()
        plot_heatmap('select', counteval, mean, b, d, g=g_raw)
        time.sleep(0.5)


def test_rannk_mu_and_rank_one_update():

    features = 2
    step_size = 1.0
    epochs = 1e3 * features ** 2

    # selection settings
    samples = 4 + floor(3 * log(features))
    mu = samples / 2
    weights = log(mu + 0.5) + torch.linspace(start=1, end=mu, steps=floor(mu)).log()
    weights = torch.flip(weights, dims=(0,)) / weights.sum()
    mu = floor(mu)
    mueff = (weights.sum() ** 2 / (weights ** 2).sum()).item()

    '''
    cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N);
    cs = (mueff + 2) / (N + mueff + 5);
    c1 = 2 / ((N + 1.3)ˆ2+mueff);
    cmu = 2 * (mueff - 2 + 1 / mueff) / ((N + 2)ˆ2+2 * mueff / 2);
    damps = 1 + 2 * max(0, sqrt((mueff - 1) / (N + 1)) - 1) + cs;
    '''

    # adaptation settings
    #cmu = mueff / features ** 2
    cc = (4 + mueff/features) / (features+4 + 2 * mueff/features)
    cs = (mueff + 2) / (features + mueff + 5)
    # c1 = 2 / ((features + 1.3) ** 2 + mueff)
    c1 = 0.3
    # cmu = 2 * (mueff - 2 + 1 / mueff) / ((features + 2)**2 + 2 * mueff / 2)
    cmu = 0.3
    damps = 1 + 2 * max(0.0, sqrt((mueff - 1.0) / (features + 1)) -1 ) + cs

    print(f'cc : {cc}, cs: {cs}, c1: {c1}, cmu: {cmu}, damps: {damps}')

    plt.title('weights')
    plt.plot(weights)
    plt.show()

    mean = torch.zeros(features)
    b = torch.eye(features)
    d = torch.eye(features)
    c = torch.matmul(b.matmul(d), b.matmul(d).T)

    pc = torch.zeros(features)

    for counteval in range(8):

        # sample parameters
        s, z = cma_es.sample(samples, step_size, mean, b, d)

        # rank by fitness
        f = objective(s[:, 0], s[:, 1])
        g = [{'sample': s[i], 'z': z[i], 'fitness':f.item()} for i, f in enumerate(f)]
        g = sorted(g, key=lambda x: x['fitness'], reverse=True)
        g = g[0:mu]
        z = torch.stack([g['z'] for g in g])
        g = torch.stack([g['sample'] for g in g])
        plot_heatmap('sample ', counteval, mean, b, d, samples=s, g=g)

        mean_prev = mean.clone()
        c_prev = c.clone()
        g_raw = g.clone()

        mean = (g * weights.unsqueeze(1)).sum(0)
        zmean = (z * weights.unsqueeze(1)).sum(0)

        # a mind bending way to write a exponential smoothed moving average
        # zmean does not contain step size or mean, so allows us to add together
        # updates of different step sizes
        pc = (1 - cc) * pc + cc * b.matmul(d).matmul(zmean)
        # which we then combine to make a covariance matrix, from 1 (mean) datapoint!
        cov_pc = pc.unsqueeze(1).matmul(pc.unsqueeze(1).t())

        # estimate cov for all selected samples (weighted by rank)
        bdz = b.matmul(d).matmul(z.t())
        cmu_cov = torch.matmul(bdz, weights.diag_embed())
        cmu_cov = cmu_cov.matmul(bdz.t())

        c = (1.0 - c1 - cmu) * c_prev + c1 * cov_pc + cmu * cmu_cov

        # pull out the eigenthings and do the business
        d, b = torch.symeig(c, eigenvectors=True)
        d = d.sqrt().diag_embed()
        plot_heatmap('select', counteval, mean, b, d, g=g_raw)
        time.sleep(0.5)
