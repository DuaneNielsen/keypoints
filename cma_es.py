import gym
from models import transporter
from models import functional as KF
import torch
import config
from tqdm import tqdm
import datasets as ds
from torchvision.transforms import functional as TVF
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from utils import UniImageViewer, plot_keypoints_on_image
from torch import nn
from torch.nn.functional import softmax
from higgham import isPD, np_nearestPD


def get_action(s, prepro, transform, view, policy, action_map):
    s = prepro(s)
    s_t = transform(s).unsqueeze(0).type(policy.dtype).to(args.device)
    kp = view(s_t)
    p = softmax(kp.flatten().matmul(policy), dim=0)
    a = Categorical(p).sample()
    a = action_map(a)
    return a, kp


def evaluate(args, env, policy, render=False):
    with torch.no_grad():
        v = UniImageViewer()

        datapack = ds.datasets[args.dataset]

        if args.model_type != 'nop':

            transporter_net = transporter.make(args).to(args.device)
            view = Keypoints(transporter_net)

        else:
            view = nop

        s = env.reset()
        a, kp = get_action(s, datapack.prepro, datapack.transforms, view, policy, datapack.action_map)

        done = False
        reward = 0.0

        while not done:
            s, r, done, i = env.step(a)
            reward += r
            a, kp = get_action(s, datapack.prepro, datapack.transforms, view, policy, datapack.action_map)
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


if __name__ == '__main__':

    args = config.config()

    if args.model_keypoints:
        policy_features = args.model_keypoints * 2
    else:
        policy_features = args.policy_inputs

    datapack = ds.datasets[args.dataset]
    env = gym.make(datapack.env)
    actions = datapack.action_map.size

    # need 2 times the number of samples or covariance matrix might not be positive semi definite
    # meaning at least one x will be linear in the other
    num_candidates = 8
    size = policy_features * actions
    m = torch.zeros(size, device=args.device, dtype=torch.float)
    step = torch.ones(size, device=args.device, dtype=torch.float)
    #step += torch.randn_like(step) * 0.1
    c = torch.eye(size, device=args.device)

    best_reward = -50000.0

    for _ in range(500):

        dist = MultivariateNormal(m, c)
        candidates = dist.sample((num_candidates,)).to(args.device)
        generation = []
        for weights in tqdm(candidates):
            policy = weights.reshape(policy_features, actions)
            reward = evaluate(args, env, policy)
            generation.append({'weights': weights, 'reward': reward})

        # get the fittest half
        generation = sorted(generation, key=lambda x: x['reward'], reverse=True)
        print([candidate['reward'] for candidate in generation])

        # if this is the best, save and show it
        best_of_generation = generation[0]
        if best_of_generation['reward'] > best_reward:
            torch.save(best_of_generation, 'best_of_generation.pt')
            best_reward = best_of_generation['reward']
            policy = best_of_generation['weights'].reshape(policy_features, actions)
            evaluate(args, env, policy, args.display)

        g = torch.stack([candidate['weights'] for candidate in generation])
        g = g[0:num_candidates // 2]

        # compute new mean and covariance matrix
        m_p = m.clone()
        c_p = c.clone()

        m = g.mean(0)
        g = g - m
        c = (g.T.matmul(g) / g.size(0))

        if not isPD(c):
            c_np = c.cpu().numpy()
            print(c_np)
            c_np = np_nearestPD(c_np)
            c = torch.from_numpy(c_np).to(dtype=c.dtype).to(args.device)

        # try:
        #     torch.cholesky(c)
        # except Exception:
        #     # fix it so we have positive definite matrix
        #     # could also use the Higham algorithm for more accuracy
        #     #  N.J. Higham, "Computing a nearest symmetric positive semidefinite
        #     # https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd
        #     print('covariance matrix not positive definite, attempting recovery')
        #     e, v = torch.symeig(c, eigenvectors=True)
        #     eps = 1e-6
        #     e[e < eps] = eps
        #     c = torch.matmul(v, torch.matmul(e.diag_embed(), v.T))
        #     try:
        #         torch.cholesky(c)
        #     except Exception:
        #         print('covariance matrix not positive definite, discarding run')
        #         m = m_p
        #         c = c_p
        #


