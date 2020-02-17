from matplotlib import pyplot as plt
import torch
import torch.nn as nn

import cma_es
import main
from matplotlib.patches import Ellipse, Circle
from math import acos, degrees, log, floor, sqrt
import time
from torch.distributions import MultivariateNormal
from keypoints import models as knn

objective_mean = (1.0, 1.0)


def spike(x, y):
    return 1 / ((x * 2.0 + objective_mean[0]) ** 2 + (y * 2.0 + objective_mean[1]) ** 2).sqrt()


def plot_objective(objective):
    x_, y_, = torch.linspace(-1, 1, 100), torch.linspace(-1, 1, 100)
    x, y = torch.meshgrid([x_, y_])
    z = objective(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x.cpu().numpy(), y.cpu().numpy(), z.cpu().numpy())
    plt.show()
    plt.imshow(z, cmap='hot')
    plt.show()


def test_plot_spike():
    plot_objective(spike)


def plot_heatmap(title, count, mean, b, d, step_size=1.0, samples=None, g=None, chiN=None):
    axis_scale = 1.2
    x_, y_, = torch.linspace(-1, 1, 100), torch.linspace(-1, 1, 100)
    x, y = torch.meshgrid([x_, y_])
    z = spike(x, y)
    fig = plt.figure()

    if step_size is not None:
        fig.suptitle(f'{title} {count} step_size: {step_size}', fontsize=16)
    else:
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

    covar = Ellipse(xy=(mean[0], mean[1]), width=d[0, 0] * 2 * step_size, height=d[1, 1] * 2 * step_size, angle=-degrees(theta), alpha=0.2)
    ax2.add_artist(covar)

    if chiN is not None:
        radius = b.matmul(d.matmul(torch.eye(2) * chiN))[0, 0]
        expected_norm = Circle(xy=(mean[0], mean[1]), radius=radius, alpha=0.1, color='yellow')
        ax2.add_artist(expected_norm)

    max_g_x = g[:, 0].abs().max() if g is not None else 0
    max_g_y = g[:, 1].abs().max() if g is not None else 0
    max_s_x = samples[:, 0].abs().max() if samples is not None else 0
    max_s_y = samples[:, 1].abs().max() if samples is not None else 0

    xscale = max(xunit_x.abs().max().item(), yunit_x.abs().max().item(), 0.3, max_g_x, max_s_x, objective_mean[0]) * axis_scale
    yscale = max(xunit_y.abs().max().item(), yunit_y.abs().max().item(), 0.4, max_g_y, max_s_y, objective_mean[1]) * axis_scale

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
        f = spike(s[:, 0], s[:, 1])
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
        f = spike(s[:, 0], s[:, 1])
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


def expect_multivariate_norm(N):
    return N ** 0.5 * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))


def test_expectation_multivariate_norm():

    xrange = range(1, 50)
    analyticE = [ expect_multivariate_norm(N) for N in range(1, 50)]

    E = []
    for N in xrange:
        dist = MultivariateNormal(torch.zeros(N), torch.eye(N))
        norms = []
        for sample in dist.sample((100,)).unbind(0):
            norms.append(sample.norm().item())
        E.append(sum(norms)/100)

    plt.plot(xrange, E)
    plt.plot(xrange, analyticE)
    plt.show()


def test_expectation_my_multivariate_norm():

    xrange = range(1, 50)
    analyticE = [ expect_multivariate_norm(N) for N in range(1, 50)]

    E = []
    for N in xrange:
        s, z = cma_es.sample(20, 1.0, torch.zeros(N), torch.eye(N), torch.eye(N))
        E.append(sum([n.norm().item() for n in z.unbind(0)])/20)

    plt.plot(xrange, E, label='empirical')
    plt.plot(xrange, analyticE, label='analytic')
    plt.legend(loc='upper  left')
    plt.show()


def test_rank_mu_and_rank_one_update():

    features = 2
    step_size = 1.0
    epochs = 1e3 * features ** 2

    # selection settings
    samples = 4 + floor(3 * log(features))
    mu = samples / 2
    weights = log(mu + 0.5) + torch.linspace(start=1, end=mu, steps=floor(mu)).log()
    weights = weights / weights.sum()
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
    damps = 1.0
    chiN = expect_multivariate_norm(features)

    print(f'cc : {cc}, cs: {cs}, c1: {c1}, cmu: {cmu}, damps: {damps}, chiN:{chiN}')


    plt.title('weights')
    plt.plot(weights)
    plt.show()

    mean = torch.zeros(features)
    b = torch.eye(features)
    d = torch.eye(features)
    c = torch.matmul(b.matmul(d), b.matmul(d).T)

    pc = torch.zeros(features)
    ps = torch.zeros(features)

    for counteval in range(8):

        # sample parameters
        s, z = cma_es.sample(samples, step_size, mean, b, d)

        # rank by fitness
        f = spike(s[:, 0], s[:, 1])
        g = [{'sample': s[i], 'z': z[i], 'fitness':f.item()} for i, f in enumerate(f)]
        g = sorted(g, key=lambda x: x['fitness'], reverse=True)
        g = g[0:mu]
        z = torch.stack([g['z'] for g in g])
        g = torch.stack([g['sample'] for g in g])
        plot_heatmap('sample ', counteval, mean, b, d, samples=s, g=g)

        # backup
        mean_prev = mean.clone()
        c_prev = c.clone()
        g_raw = g.clone()

        mean = (g * weights.unsqueeze(1)).sum(0)
        zmean = (z * weights.unsqueeze(1)).sum(0)

        # step size
        ps = (1 - cs) * ps + cs * b.matmul(zmean)
        step_size = step_size * (cs * ps.norm() / chiN - 1.0).exp()

        # a mind bending way to write a exponential smoothed moving average
        # zmean does not contain step size or mean, so allows us to add together
        # updates of different step sizes
        pc = (1 - cc) * pc + cc * b.matmul(d).matmul(zmean)
        # which we then combine to make a covariance matrix, from 1 (mean) datapoint!
        # this is why it's called "rank 1" update
        cov_pc = pc.unsqueeze(1).matmul(pc.unsqueeze(1).t())

        # estimate cov for all selected samples (weighted by rank)
        bdz = b.matmul(d).matmul(z.t())
        cmu_cov = torch.matmul(bdz, weights.diag_embed())
        cmu_cov = cmu_cov.matmul(bdz.t())

        c = (1.0 - c1 - cmu) * c_prev + c1 * cov_pc + cmu * cmu_cov

        # pull out the eigenthings and do the business
        d, b = torch.symeig(c, eigenvectors=True)
        d = d.sqrt().diag_embed()
        plot_heatmap('select', counteval, mean, b, d, g=g_raw, chiN=chiN)
        time.sleep(0.5)


def test_rank_mu_and_rank_one_update_with_step_size_control():

    features = 2
    step_size = 1.0
    epochs = 1e3 * features ** 2

    # selection settings
    samples = 4 + floor(3 * log(features))
    mu = samples / 2
    weights = log(mu + 0.5) + torch.linspace(start=1, end=mu, steps=floor(mu)).log()
    weights = weights / weights.sum()
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
    # cs = (mueff + 2) / (features + mueff + 5)
    cs = 0.95
    # c1 = 2 / ((features + 1.3) ** 2 + mueff)
    c1 = 0.3
    # cmu = 2 * (mueff - 2 + 1 / mueff) / ((features + 2)**2 + 2 * mueff / 2)
    cmu = 0.3
    damps = 1 + 2 * max(0.0, sqrt((mueff - 1.0) / (features + 1)) -1) + cs
    damps = 1.0
    chiN = expect_multivariate_norm(features)

    print(f'cc : {cc}, cs: {cs}, c1: {c1}, cmu: {cmu}, damps: {damps}, chiN:{chiN}')


    plt.title('weights')
    plt.plot(weights)
    plt.show()

    mean = torch.zeros(features)
    b = torch.eye(features)
    d = torch.eye(features)
    c = torch.matmul(b.matmul(d), b.matmul(d).T)

    pc = torch.zeros(features)
    ps = torch.zeros(features)

    for counteval in range(8):

        # sample parameters
        s, z = cma_es.sample(samples, step_size, mean, b, d)

        # rank by fitness
        f = spike(s[:, 0], s[:, 1])
        g = [{'sample': s[i], 'z': z[i], 'fitness':f.item()} for i, f in enumerate(f)]
        g = sorted(g, key=lambda x: x['fitness'], reverse=True)
        g = g[0:mu]
        z = torch.stack([g['z'] for g in g])
        g = torch.stack([g['sample'] for g in g])
        plot_heatmap('sample ', counteval, mean, b, d, samples=s, g=g, chiN=chiN)

        # backup
        mean_prev = mean.clone()
        c_prev = c.clone()
        g_raw = g.clone()

        mean = (g * weights.unsqueeze(1)).sum(0)
        zmean = (z * weights.unsqueeze(1)).sum(0)

        # step size
        ps = (1 - cs) * ps + cs * b.matmul(zmean)
        step_size = step_size * ((cs / damps) * (ps.norm() / chiN - 1.0)).exp()

        # a mind bending way to write a exponential smoothed moving average
        # zmean does not contain step size or mean, so allows us to add together
        # updates of different step sizes
        pc = (1 - cc) * pc + cc * b.matmul(d).matmul(zmean)
        # which we then combine to make a covariance matrix, from 1 (mean) datapoint!
        # this is why it's called "rank 1" update
        cov_pc = pc.unsqueeze(1).matmul(pc.unsqueeze(1).t())

        # estimate cov for all selected samples (weighted by rank)
        bdz = b.matmul(d).matmul(z.t())
        cmu_cov = torch.matmul(bdz, weights.diag_embed())
        cmu_cov = cmu_cov.matmul(bdz.t())

        c = (1.0 - c1 - cmu) * c_prev + c1 * cov_pc + cmu * cmu_cov

        # pull out the eigenthings and do the business
        d, b = torch.symeig(c, eigenvectors=True)
        d = d.sqrt().diag_embed()
        plot_heatmap('select', counteval, mean, b, d, g=g_raw, chiN=chiN, step_size=step_size)
        time.sleep(0.5)

from math import pi, e


def akley(x, y):
    x, y = x + objective_mean[0], y + objective_mean[1]
    return 20 * torch.exp(-0.2 * torch.sqrt(0.5 * (x**2 + y **2))) - \
           torch.exp(0.5 * (torch.cos(2*pi*x) + torch.cos(2*pi*y))) + e + 20

def test_plot_akley():
    plot_objective(akley)


def test_hyperparams():

    objective_f = akley

    features = 2
    step_size = 1.0
    epochs = 1e3 * features ** 2

    # selection settings
    samples = 4 + floor(3 * log(features))
    mu = samples / 2
    weights = torch.tensor([log(mu + 0.5)]) - torch.linspace(start=1, end=mu, steps=floor(mu)).log()
    weights = weights / weights.sum()
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
    cc = (4 + mueff/features) / (features+4 + 2 * mueff/features)
    cs = (mueff + 2) / (features + mueff + 5)
    c1 = 2 / ((features + 1.3) ** 2 + mueff)
    cmu = 2 * (mueff - 2 + 1 / mueff) / ((features + 2)**2 + 2 * mueff / 2)
    damps = 1 + 2 * max(0.0, sqrt((mueff - 1.0) / (features + 1)) -1) + cs
    chiN = expect_multivariate_norm(features)

    mean = torch.zeros(features)
    b = torch.eye(features)
    d = torch.eye(features)
    c = torch.matmul(b.matmul(d), b.matmul(d).T)

    pc = torch.zeros(features)
    ps = torch.zeros(features)

    print(f'mu: {mu}. mueff: {mueff}, cc : {cc}, cs: {cs}, c1: {c1}, cmu: {cmu}, damps: {damps}, chiN:{chiN}')

    plt.title('weights')
    plt.plot(weights)
    print(weights)
    plt.show()
    step_size_l = [step_size]
    correlation_l = [1.0]
    ps_l = [ps[0].item()]
    fitness_l = [0]
    plot_freq = 1

    for counteval in range(1, 10):

        # sample parameters
        s, z = cma_es.sample(samples, step_size, mean, b, d)

        # rank by fitness
        f = objective_f(s[:, 0], s[:, 1])
        g = [{'sample': s[i], 'z': z[i], 'fitness':f.item()} for i, f in enumerate(f)]
        g = sorted(g, key=lambda x: x['fitness'], reverse=True)
        g = g[0:mu]
        fitness_l.append(g[0]['fitness'])
        z = torch.stack([g['z'] for g in g])
        g = torch.stack([g['sample'] for g in g])

        if counteval % plot_freq == 0:
            plot_heatmap('sample ', counteval, mean, b, d, samples=s, g=g, chiN=chiN, step_size=step_size)

        # backup
        mean_prev = mean.clone()
        prev_cov = c.clone()
        g_raw = g.clone()

        mean = (g * weights.unsqueeze(1)).sum(0)
        zmean = (z * weights.unsqueeze(1)).sum(0)

        # step size
        ps = (1 - cs) * ps + sqrt(cs * (2.0 - cs)) * b.matmul(zmean)

        correlation = ps.norm() / chiN
        ps_l.append(ps[0].item())
        correlation_l.append(correlation.item())

        # delay the introduction of the rank 1 update
        denominator = sqrt(1 - (1 - cs) ** (2 * counteval / samples))
        threshold = 1.4e2 / features + 1
        hsig = correlation / denominator < threshold
        hsig = 1.0 if hsig else 0.0

        #step_size = step_size * ((cs / damps) * (correlation - 1.0)).exp()
        step_size = step_size * ((cs/damps) * (correlation - 1.0)).exp()

        step_size_l.append(step_size)

        # a mind bending way to write a exponential smoothed moving average
        # zmean does not contain step size or mean, so allows us to add together
        # updates of different step sizes
        pc = (1 - cc) * pc + hsig * sqrt(cc*(2.0 - cc)*mueff) * b.matmul(d).matmul(zmean)
        # which we then combine to make a covariance matrix, from 1 (mean) datapoint!
        # this is why it's called "rank 1" update
        pc_cov = pc.unsqueeze(1).matmul(pc.unsqueeze(1).t())
        # mix back in the old covariance if hsig == 0
        pc_cov = pc_cov + (1 - hsig) * cc * (2 - cc) * prev_cov

        # estimate cov for all selected samples (weighted by rank)
        bdz = b.matmul(d).matmul(z.t())
        cmu_cov = torch.matmul(bdz, weights.diag_embed())
        cmu_cov = cmu_cov.matmul(bdz.t())

        c = (1.0 - c1 - cmu) * prev_cov + (c1 * pc_cov) + (cmu * cmu_cov)

        # pull out the eigenthings and do the business
        d, b = torch.symeig(c, eigenvectors=True)
        d = d.sqrt().diag_embed()
        if counteval % plot_freq == 0:
            plot_heatmap('select', counteval, mean, b, d, g=g_raw, chiN=chiN, step_size=step_size)


def spike_one(w):
    x, y = w[:, 0], w[:, 1]
    return 1 / ((x * 2.0 + objective_mean[0]) ** 2 + (y * 2.0 + objective_mean[1]) ** 2).sqrt()


def test_FastCMA():

    fast_cma = cma_es.FastCovarianceMatrixAdaptation(2)

    metrics = {'step_size': [], 'correlation': [], 'fitness_max': []}

    for _ in range(1, 20):
        ranked_results, info = fast_cma.step(spike_one)
        selected = torch.stack([g['parameters'] for g in ranked_results[0:fast_cma.mu]])
        plot_heatmap('select', fast_cma.gen_count, fast_cma.mean, fast_cma.b, fast_cma.d,
                     g=selected, chiN=fast_cma.chiN, step_size=fast_cma.step_size)
        for key in metrics:
            metrics[key].append(info[key])

    for key in metrics:
        plt.title(key)
        plt.plot(metrics[key])
        plt.legend(loc='lower left')
        plt.show()


def test_NaiveCMA():

    n_cma = cma_es.NaiveCovarianceMatrixAdaptation(2)

    metrics = {'fitness_mean': [], 'fitness_max': [], 'c_norm': []}

    for gen in range(1, 20):
        ranked_results, info = n_cma.step(spike_one)
        selected = torch.stack([g['parameters'] for g in ranked_results[0:n_cma.mu]])
        d, b = torch.symeig(n_cma.c, eigenvectors=True)
        d = d.sqrt().diag_embed()
        plot_heatmap('select', gen, n_cma.mean, b, d, g=selected)
        for key in metrics:
            metrics[key].append(info[key])

    for key in metrics:
        plt.title(key)
        plt.plot(metrics[key])
        plt.legend(loc='lower left')
        plt.show()


def test_modularity():

    policy = nn.Linear(1, 1)

    weights = torch.randn(knn.parameter_count(policy))

    policy = knn.load_weights(policy, weights)
    policy_weights = knn.flatten(policy)

    print(weights)
    print(policy_weights)
    assert torch.allclose(weights, policy_weights)