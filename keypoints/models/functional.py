import torch
import torch.nn.functional as F


def marginal_softmax(heatmap, dim):
    marginal = torch.mean(heatmap, dim=dim)
    sm = F.softmax(marginal, dim=2)
    return sm


def marginal_logsoftmax(heatmap, dim):
    marginal = torch.mean(heatmap, dim=dim)
    sm = F.log_softmax(marginal, dim=2)
    return sm


def prob_to_keypoints(prob, length):
    ruler = torch.linspace(0, 1, length).type_as(prob).expand(1, 1, -1).to(prob.device)
    return torch.sum(prob * ruler, dim=2, keepdim=True).squeeze(2)


def logprob_to_keypoints(prob, length):
    ruler = torch.log(torch.linspace(0, 1, length, device=prob.device)).type_as(prob).expand(1, 1, -1)
    return torch.sum(torch.exp(prob + ruler), dim=2, keepdim=True).squeeze(2)


def spacial_softmax(heatmap, probs=False):
    height, width = heatmap.size(2), heatmap.size(3)
    hp, wp = marginal_softmax(heatmap, dim=3), marginal_softmax(heatmap, dim=2)
    hk, wk = prob_to_keypoints(hp, height), prob_to_keypoints(wp, width)
    if probs:
        return torch.stack((hk, wk), dim=2), (hp, wp)
    else:
        return torch.stack((hk, wk), dim=2)


def spacial_logsoftmax(heatmap, probs=False):
    height, width = heatmap.size(2), heatmap.size(3)
    hp, wp = marginal_logsoftmax(heatmap, dim=3), marginal_logsoftmax(heatmap, dim=2)
    hk, wk = logprob_to_keypoints(hp, height), logprob_to_keypoints(wp, width)
    if probs:
        return torch.stack((hk, wk), dim=2), (torch.exp(hp), torch.exp(wp))
    else:
        return torch.stack((hk, wk), dim=2)




def squared_diff(h, height):
    hs = torch.linspace(0, 1, height, device=h.device).type_as(h).expand(h.shape[0], h.shape[1], height)
    hm = h.expand(height, -1, -1).permute(1, 2, 0)
    hm = ((hs - hm) ** 2)
    return hm


def gaussian_like_function(kp, height, width, sigma=0.1, eps=1e-6):
    hm = squared_diff(kp[:, :, 0], height)
    wm = squared_diff(kp[:, :, 1], width)
    hm = hm.expand(width, -1, -1, -1).permute(1, 2, 3, 0)
    wm = wm.expand(height, -1, -1, -1).permute(1, 2, 0, 3)
    gm = - (hm + wm + eps).sqrt_() / (2 * sigma ** 2)
    gm = torch.exp(gm)
    return gm


def point_map(kp, h, w):
    hm = squared_diff(kp[:, :, 0], h)
    wm = squared_diff(kp[:, :, 1], w)
    hm = hm.expand(w, -1, -1, -1).permute(1, 2, 3, 0)
    wm = wm.expand(h, -1, -1, -1).permute(1, 2, 0, 3)
    gm = - (hm + wm)
    return gm


# def patch_axis(c, l, w):
#     lower = c - w/l/2.0
#     upper = c + w/l/2.0
#     lower = lower.clamp(0.0, 1.0)
#     upper = upper.clamp(0.0, 1.0)
#     lower = torch.floor(lower * l)
#     upper = torch.floor(upper * l)
#     x = torch.zeros(c.size(0), l)
#     x[:, lower:upper] = 1.0
#     return x


def patch_axis(c, l, w):
    x = torch.linspace(0, l, l)
    y = (2.0/w) * x + c
    return x, y



""" NOT DIFFERENTIABLE """
def patch_map(kp, h, w, ph, pw):
    ax1 = patch_axis(kp[:, :, 0], h, w)

