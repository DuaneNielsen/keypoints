import gym
from models import transporter
from models import functional as KF
import torch
import config
import datasets as ds
from torchvision.transforms import functional as TVF
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from utils import UniImageViewer, plot_keypoints_on_image
from torch import nn
from torch.nn.functional import softmax
from higgham import isPD, np_nearestPD
import multiprocessing as mp
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from statistics import mean
from pathlib import Path
from numpy import linalg as la


def make_args(args, datapack, weights, policy_features, actions, render=False):
    args_dict = {}
    args_dict['args'] = args
    args_dict['env'] = datapack.env
    args_dict['policy'] = weights.reshape(policy_features, actions).cpu().numpy()
    args_dict['render'] = render
    return args_dict


def multi_evaluate(arg_dict):
    args = arg_dict['args']
    env = gym.make(arg_dict['env'])
    policy = torch.from_numpy(arg_dict['policy']).to(args.device)
    render = arg_dict['render']
    return evaluate(args, env, policy, render)


def evaluate(args, env, policy, render=False):
    with torch.no_grad():

        def get_action(s, prepro, transform, view, policy, action_map, device):
            s = prepro(s)
            s_t = transform(s).unsqueeze(0).type(policy.dtype).to(device)
            kp = view(s_t)
            p = softmax(kp.flatten().matmul(policy), dim=0)
            a = Categorical(p).sample()
            a = action_map(a)
            return a, kp

        v = UniImageViewer()

        datapack = ds.datasets[args.dataset]

        if args.model_type != 'nop':

            transporter_net = transporter.make(args, map_device='cpu')
            view = Keypoints(transporter_net).to(args.device)

        else:
            view = nop

        s = env.reset()
        a, kp = get_action(s, datapack.prepro, datapack.transforms, view, policy, datapack.action_map, args.device)

        done = False
        reward = 0.0

        while not done:
            s, r, done, i = env.step(a)
            reward += r

            a, kp = get_action(s, datapack.prepro, datapack.transforms, view, policy, datapack.action_map, args.device)
            if render:
                if args.model_keypoints:
                    s = datapack.prepro(s)
                    s = TVF.to_tensor(s).unsqueeze(0)
                    s = plot_keypoints_on_image(kp[0], s[0])
                    v.render(s)
                else:
                   env.render()

    return reward


class Keypoints(nn.Module):
    def __init__(self, transporter_net):
        super().__init__()
        self.transporter_net = transporter_net

    def forward(self, s_t):
        heatmap = self.transporter_net.keypoint(s_t)
        kp = KF.spacial_logsoftmax(heatmap)
        return kp


def nop(s_t):
    return s_t


def sample(n, sigma, mean, B, D):
    features = mean.size(0)
    z = torch.randn(features, n, device=mean.device, dtype=mean.dtype)
    s = mean.view(-1, 1) + sigma * B.matmul(D.matmul(z))
    return s.T, z.T


def save_and_monitor():
    global best_reward
    print([candidate['reward'] for candidate in generation])

    # if this is the best, save and show it
    best_of_generation = generation[0]
    tb.add_scalar('gen/gen_mean', mean(results), global_step)
    tb.add_scalar('gen/gen_best', best_of_generation['reward'], global_step)
    tb.add_scalar('gen/gen_mean_selected', mean(results[0:num_candidates // 4]), global_step)
    if best_of_generation['reward'] > best_reward:
        torch.save(best_of_generation, log_dir + 'best_of_generation.pt')
        best_reward = best_of_generation['reward']
        policy = best_of_generation['weights'].reshape(policy_features, actions).to(args.device)
        evaluate(args, env, policy, args.display)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    """ CMA - ES algorithm

    implemented from
    
    https: // arxiv.org / abs / 1604.00772

    """""

    args = config.config()

    log_dir = f'data/cma_es/{args.run_id}/'
    tb = SummaryWriter(log_dir)
    global_step = 0

    sigma = 1.0

    if args.model_keypoints:
        policy_features = args.model_keypoints * 2
    else:
        policy_features = args.policy_inputs

    datapack = ds.datasets[args.dataset]
    env = gym.make(datapack.env)
    actions = datapack.action_map.size

    # need 2 times the number of samples or covariance matrix might not be positive semi definite
    # meaning at least one x will be linear in the other
    num_candidates = 16
    size = policy_features * actions
    m = torch.zeros(size, device=args.device)
    step = torch.ones(size, device=args.device)
    c = torch.eye(size, device=args.device)
    b = torch.eye(size, device=args.device)  # rotation covariance
    d = torch.eye(size, device=args.device)  # diagonal std-dev covariance

    best_reward = -50000.0

    if args.demo:
        while True:
            policy = torch.load(log_dir + 'best_of_generation.pt')['weights'].reshape(policy_features, actions)
            evaluate(args, env, policy, True)

    for _ in trange(args.epochs):

        candidates, z = sample(num_candidates, sigma, m, b, d)
        weights = torch.unbind(candidates, dim=0)

        worker_args = [make_args(args, datapack, w, policy_features, actions, False) for w in weights]

        with mp.Pool(processes=args.processes) as pool:
            results = pool.map(multi_evaluate, worker_args)

        generation = []
        for i in range(len(results)):
            generation.append({'weights': weights[i], 'z': z[i], 'reward': results[i]})

        # get the fittest 25 %
        generation = sorted(generation, key=lambda x: x['reward'], reverse=True)

        save_and_monitor()

        #g = torch.stack([candidate['weights'] for candidate in generation])
        g = torch.stack([candidate['z'] for candidate in generation])
        g = g[0:num_candidates // 4].to(args.device)

        # compute new mean and covariance matrix
        m_p = m.clone().to(args.device)
        c_p = c.clone().to(args.device)

        m = g.mean(0)
        g = g - m_p
        c = g.T.matmul(g) / g.size(0)
        #rank_mu_term = torch.matmul(b, torch.matmul(d, z.T))
        #c = rank_mu_term.T.matmul(rank_mu_term)
        covariance_discount = num_candidates // 4 / m.size(0) ** 2
        c = (1 - covariance_discount) * c_p + covariance_discount * c

        #d, b = la.eigh(c.cpu().numpy())
        #d, b = torch.from_numpy(d).to(args.device), torch.from_numpy(b).to(args.device)

        # decompose covariance matrix into eigevectors and rescale to stdev
        #d, b = c.symeig(True)
        #d = d.sqrt().diag_embed()

        global_step += 1




