import torch


def prob_to_keypoints(prob, length):
    ruler = torch.linspace(0, 1, length).type_as(prob).expand(1, 1, -1).to(prob.device)
    return torch.sum(prob * ruler, dim=2, keepdim=True).squeeze(2)


def calc_mean(eps, n):
    p = torch.ones(n) * eps
    p[n-1] = 1.0 - (n-1) * eps
    #assert sum(p) == 1.0
    return prob_to_keypoints(p, n)


def test_prob_2_k():
    print(calc_mean(0.001, 16))
    print(calc_mean(0.001, 32))
    print(calc_mean(0.001, 64))
    print(calc_mean(0.001, 128))

