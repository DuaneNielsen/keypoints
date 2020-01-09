import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from math import floor, sqrt, log
import config
from torch import softmax
from torch.distributions import Categorical, MultivariateNormal
import torchvision.transforms.functional as TVF
from utils import UniImageViewer, plot_keypoints_on_image
from models import transporter
from models import functional as KF
import datasets as ds
import gym
import gym_wrappers
import multiprocessing as mp


def nop(s_t):
    return s_t


class Keypoints(nn.Module):
    def __init__(self, transporter_net):
        super().__init__()
        self.transporter_net = transporter_net

    def forward(self, s_t):
        heatmap = self.transporter_net.keypoint(s_t)
        kp = KF.spacial_logsoftmax(heatmap)
        return kp


class EvalPacket():
    def __init__(self, args, datapack, weights, features, render):
        """
        Serializable arguments for eval function
        :param args:
        :param datapack:
        :param weights:
        :param render:
        """
        self.args = args
        self.datapack = datapack
        self.weights = weights
        self.render = render
        self.features = features


def encode(args, datapack, weights, features, render):
    weights = weights.cpu().numpy()
    return EvalPacket(args, datapack, weights, features, render)


def decode(packet):
    packet.weights = torch.from_numpy(packet.weights).to(packet.args.device)
    return packet


def evaluate(packet):
    packet = decode(packet)
    args = packet.args
    features = packet.features
    render = packet.render

    datapack = ds.datasets[args.dataset]
    env = gym.make(datapack.env)
    if args.gym_reward_count_limit is not None:
        env = gym_wrappers.RewardCountLimit(env, args.gym_reward_count_limit)

    actions = datapack.action_map.size
    policy = packet.weights.reshape(features, actions)

    with torch.no_grad():

        def get_action(s, prepro, transform, view, policy, action_map, device, action_select_mode='argmax'):
            s = prepro(s)
            s_t = transform(s).unsqueeze(0).type(policy.dtype).to(device)
            kp = view(s_t)
            p = softmax(kp.flatten().matmul(policy), dim=0)
            if action_select_mode == 'argmax':
                a = torch.argmax(p)
            if action_select_mode == 'sample':
                a = Categorical(p).sample()
            a = action_map(a)
            return a, kp

        v = UniImageViewer()

        if args.model_type != 'nop':
            transporter_net = transporter.make(args, map_device='cpu')
            view = Keypoints(transporter_net).to(args.device)

        else:
            view = nop

        s = env.reset()
        a, kp = get_action(s, datapack.prepro, datapack.transforms, view, policy, datapack.action_map, args.device,
                           action_select_mode=args.policy_action_select_mode)

        done = False
        reward = 0.0

        while not done:
            s, r, done, i = env.step(a)
            reward += r

            a, kp = get_action(s, datapack.prepro, datapack.transforms, view, policy, datapack.action_map, args.device,
                               action_select_mode=args.policy_action_select_mode)
            if render:
                if args.model_keypoints:
                    s = datapack.prepro(s)
                    s = TVF.to_tensor(s).unsqueeze(0)
                    s = plot_keypoints_on_image(kp[0], s[0])
                    v.render(s)
                else:
                    env.render()

    return reward


class AtariMpEvaluator(object):
    def __init__(self, args, datapack, policy_features, policy_actions, render=False):
        self.args = args
        self.datapack = datapack
        self.policy_features = policy_features
        self.policy_actions = policy_actions
        self.render = render

    def fitness(self, candidates):
        weights = torch.unbind(candidates, dim=0)

        worker_args = [encode(self.args, self.datapack, w, self.policy_features, self.render) for w in weights]

        with mp.Pool(processes=args.processes) as pool:
            results = pool.map(evaluate, worker_args)

        results = torch.tensor(results)
        return results

    def len_policy_weights(self):
        return self.policy_features * self.policy_actions


def sample(n, sigma, mean, B, D):
    features = mean.size(0)
    z = torch.randn(features, n, device=mean.device, dtype=mean.dtype)
    s = mean.view(-1, 1) + sigma * B.matmul(D.matmul(z))
    return s.T, z.T


def simple_sample(features, n, mean, c):
    z = torch.randn(features, n, device=mean.device, dtype=mean.dtype)
    s = mean.view(-1, 1) + c.matmul(z)
    return s.T


def expect_multivariate_norm(N):
    return N ** 0.5 * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))


class CMA(object):
    def _rank(self, results, rank_order='max'):
        if rank_order == 'max':
            ranked_results = sorted(results, key=lambda x: x['fitness'], reverse=True)
        elif rank_order == 'min':
            ranked_results = sorted(results, key=lambda x: x['fitness'])
        else:
            raise Exception(f'invalid value for kwarg type {rank_order}, valid values are max or min')

        return ranked_results

    def step(self, object_f, rank_order='max'):
        pass


class NaiveCovarianceMatrixAdaptation(CMA):
    def __init__(self, N, cma=None, samples=None):
        self.N = N
        self.recommended_steps = range(1, floor(1e3 * N ** 2))
        # variables
        self.mean = torch.zeros(N)
        self.c = torch.eye(N)
        self.samples = 4 + floor(3 * log(N)) * 2 if samples is None else samples
        self.mu = self.samples // 2
        self.gen_count = 0
        self.cmu = self.mu / N ** 2 if cma is None else cma

    def step(self, objective_f, rank_order='max'):
        params = simple_sample(self.N, self.samples, self.mean, self.c)
        # rank by fitness
        f = objective_f(params)
        results = [{'parameters': params[i], 'fitness': f.item()} for i, f in enumerate(f)]
        ranked_results = self._rank(results, rank_order)

        selected_results = ranked_results[0:self.mu]
        g = torch.stack([g['parameters'] for g in selected_results])

        mean_prev = self.mean.clone()
        self.mean = g.mean(0)
        g = g - mean_prev
        c_cma = torch.matmul(g.T, g) / self.mu
        self.c = (1 - self.cmu) * self.c + self.cmu * c_cma

        info = {'fitness_max': f.max(), 'fitness_mean': f.mean(), 'c_norm': self.c.norm()}
        self.gen_count += 1

        return ranked_results, info

    def __repr__(self):
        return f'N: {self.N}, samples: {self.samples}, mu: {self.mu}, cmu: {self.cmu}'


class SimpleCovarianceMatrixAdaptation(CMA):
    def __init__(self, N, cma=None, samples=None):
        self.N = N
        self.recommended_steps = range(1, floor(1e3 * N ** 2))

        self.samples = 4 + floor(3 * log(N)) * 2 if samples is None else samples
        self.mu = self.samples // 4
        self.gen_count = 0
        self.cmu = self.mu / N ** 2 if cma is None else cma

        # variables
        self.mean = torch.zeros(N)
        self.b = torch.eye(N)
        self.d = torch.eye(N)
        self.c = torch.matmul(self.b.matmul(self.d), self.b.matmul(self.d).T)  # c = B D D B.T

    def step(self, objective_f, rank_order='max'):

        # sample parameters
        params, z = sample(self.samples, 1.0, self.mean, self.b, self.d)

        # rank by fitness
        f = objective_f(params)
        results = [{'parameters': params[i], 'z': z[i], 'fitness': f.item()} for i, f in enumerate(f)]
        ranked_results = self._rank(results, rank_order)

        selected_results = ranked_results[0:self.mu]
        g = torch.stack([g['parameters'] for g in selected_results])
        z = torch.stack([g['z'] for g in selected_results])

        self.mean = g.mean(0)
        bdz = self.b.matmul(self.d).matmul(z.t())
        c_mu = torch.matmul(bdz, torch.eye(self.mu) / self.mu)
        c_mu = c_mu.matmul(bdz.t())

        self.c = (1 - self.cmu) * self.c + self.cmu * c_mu

        self.d, self.b = torch.symeig(self.c, eigenvectors=True)
        self.d = self.d.sqrt().diag_embed()

        info = {'fitness_max': f.max(), 'fitness_mean': f.mean(), 'c_norm': self.c.norm(), 'max_eigv': self.d.max()}
        self.gen_count += 1

        return ranked_results, info

    def __repr__(self):
        return f'N: {self.N}, samples: {self.samples}, mu: {self.mu}, cmu: {self.cmu}'


class FastCovarianceMatrixAdaptation(CMA):
    def __init__(self, N, step_mode='auto', step_decay=None, initial_step_size=None, samples=None):
        self.N = N
        self.recommended_steps = range(1, floor(1e3 * N ** 2))

        # selection settings
        self.samples = 4 + floor(3 * log(N)) if samples is None else samples
        self.mu = self.samples / 2
        self.weights = torch.tensor([log(self.mu + 0.5)]) - torch.linspace(start=1, end=self.mu,
                                                                           steps=floor(self.mu)).log()
        self.weights = self.weights / self.weights.sum()
        self.weights = self.weights / self.weights.sum()
        self.mu = floor(self.mu)
        self.mueff = (self.weights.sum() ** 2 / (self.weights ** 2).sum()).item()


        # adaptation settings
        self.cc = (4 + self.mueff / N) / (N + 4 + 2 * self.mueff / N)
        self.cs = (self.mueff + 2) / (N + self.mueff + 5)
        self.c1 = 2 / ((self.N + 1.3) ** 2 + self.mueff)
        self.cmu = 2 * (self.mueff - 2 + 1 / self.mueff) / ((N + 2) ** 2 + 2 * self.mueff / 2)
        self.damps = 1 + 2 * max(0.0, sqrt((self.mueff - 1.0) / (N + 1)) - 1) + self.cs
        self.chiN = expect_multivariate_norm(N)
        self.step_size = 0.5 if initial_step_size is None else initial_step_size
        self.step_mode = step_mode
        if step_mode == 'decay' and step_decay is None:
            raise Exception('decay mode requires you set a step decay')
        self.step_decay = (1.0 - step_decay) if step_decay is not None else None

        # variables
        self.mean = torch.zeros(N)
        self.b = torch.eye(N)
        self.d = torch.eye(N)
        self.c = torch.matmul(self.b.matmul(self.d), self.b.matmul(self.d).T)  # c = B D D B.T

        self.pc = torch.zeros(N)
        self.ps = torch.zeros(N)
        self.gen_count = 1

    def step(self, objective_f, rank_order='max'):

        # sample parameters
        s, z = sample(self.samples, self.step_size, self.mean, self.b, self.d)

        # rank by fitness
        f = objective_f(s)
        results = [{'parameters': s[i], 'z': z[i], 'fitness': f.item()} for i, f in enumerate(f)]

        if rank_order == 'max':
            ranked_results = sorted(results, key=lambda x: x['fitness'], reverse=True)
        elif rank_order == 'min':
            ranked_results = sorted(results, key=lambda x: x['fitness'])
        else:
            raise Exception(f'invalid value for kwarg type {rank_order}, valid values are max or min')

        selected_results = ranked_results[0:self.mu]
        z = torch.stack([g['z'] for g in selected_results])
        g = torch.stack([g['parameters'] for g in selected_results])

        self.mean = (g * self.weights.unsqueeze(1)).sum(0)
        zmean = (z * self.weights.unsqueeze(1)).sum(0)

        # step size
        self.ps = (1 - self.cs) * self.ps + sqrt(self.cs * (2.0 - self.cs)) * self.b.matmul(zmean)

        correlation = self.ps.norm() / self.chiN

        # delay the introduction of the rank 1 update
        denominator = sqrt(1 - (1 - self.cs) ** (2 * self.gen_count / self.samples))
        threshold = 1.4e2 / self.N + 1
        hsig = correlation / denominator < threshold
        hsig = 1.0 if hsig else 0.0

        # adapt step size
        if self.step_mode == 'auto':
            self.step_size = self.step_size * ((self.cs / self.damps) * (correlation - 1.0)).exp()
        elif self.step_mode == 'nodamp':
            self.step_size = self.step_size * (correlation - 1.0).exp()
        elif self.step_mode == 'decay':
            self.step_size = self.step_size * self.step_decay
        elif self.step_mode == 'constant':
            pass
        else:
            raise Exception('step_mode must be auto | nodamp | decay | constant')

        # a mind bending way to write a exponential smoothed moving average
        # zmean does not contain step size or mean, so allows us to add together
        # updates of different step sizes
        self.pc = (1 - self.cc) * self.pc + hsig * sqrt(self.cc * (2.0 - self.cc) * self.mueff) * self.b.matmul(
            self.d).matmul(zmean)
        # which we then combine to make a covariance matrix, from 1 (mean) datapoint!
        # this is why it's called "rank 1" update
        pc_cov = self.pc.unsqueeze(1).matmul(self.pc.unsqueeze(1).t())
        # mix back in the old covariance if hsig == 0
        pc_cov = pc_cov + (1 - hsig) * self.cc * (2 - self.cc) * self.c

        # estimate cov for all selected samples (weighted by rank)
        bdz = self.b.matmul(self.d).matmul(z.t())
        cmu_cov = torch.matmul(bdz, self.weights.diag_embed())
        cmu_cov = cmu_cov.matmul(bdz.t())

        self.c = (1.0 - self.c1 - self.cmu) * self.c + (self.c1 * pc_cov) + (self.cmu * cmu_cov)

        # pull out the eigenthings and do the business
        self.d, self.b = torch.symeig(self.c, eigenvectors=True)
        self.d = self.d.sqrt().diag_embed()
        self.gen_count += 1

        info = {'step_size': self.step_size, 'correlation': correlation,
                'fitness_max': f.max(), 'fitness_mean': f.mean(), 'c_norm': self.c.norm(),
                'max_eigv': self.d.max()}
        return ranked_results, info

    def __repr__(self):
        return f'N: {self.N}, samples: {self.samples}, mu: {self.mu}, mueff: {self.mueff}, cc: {self.cc}, ' \
               f'cs: {self.cs}, c1: {self.c1}, cmu: {self.cmu}, damps: {self.damps}, chiN: {self.chiN}, ' \
               f'step_mode: {self.step_mode}, step_decay: {self.step_decay}, step_size: {self.step_size}'


if __name__ == '__main__':

    args = config.config()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    datapack = ds.datasets[args.dataset]
    log_dir = f'data/cma_es/{datapack.env}/{args.run_id}/'
    tb = SummaryWriter(log_dir)
    global_step = 0
    best_reward = -1e8
    show = False

    if args.model_keypoints:
        policy_features = args.model_keypoints * 2
    else:
        policy_features = args.policy_inputs

    evaluator = AtariMpEvaluator(args, datapack, policy_features, datapack.action_map.size)
    demo = AtariMpEvaluator(args, datapack, policy_features, datapack.action_map.size, render=True)

    if args.cma_algo == 'fast':
        cma = FastCovarianceMatrixAdaptation(N=evaluator.len_policy_weights(),
                                             step_mode=args.cma_step_mode,
                                             step_decay=args.cma_step_decay,
                                             initial_step_size=args.cma_initial_step_size,
                                             samples=args.cma_samples)
    elif args.cma_algo == 'naive':
        cma = NaiveCovarianceMatrixAdaptation(N=evaluator.len_policy_weights(), samples=args.cma_samples)
    elif args.cma_algo == 'simple':
        cma = SimpleCovarianceMatrixAdaptation(N=evaluator.len_policy_weights(), samples=args.cma_samples)
    else:
        raise Exception('--cma_algo fast | naive | simple')

    tb.add_text('args', str(args), global_step)
    tb.add_text('cma_params', str(cma), global_step)

    for step in cma.recommended_steps:

        ranked_results, info = cma.step(evaluator.fitness)

        print([result['fitness'] for result in ranked_results])

        for key, value in info.items():
            tb.add_scalar(key, value, global_step)

        if ranked_results[0]['fitness'] > best_reward:
            best_reward = ranked_results[0]['fitness']
            torch.save(ranked_results[0]['parameters'], log_dir + 'best_of_generation.pt')
            show = True

        if args.display and (global_step % args.display_freq == 0 or show):
            demo.fitness(ranked_results[0]['parameters'].unsqueeze(0))

        show = False
        global_step += 1

        if args.epochs is not None and step >= args.epochs:
            break
