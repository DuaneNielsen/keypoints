import torch
import torch.nn.functional as F


def marginal_softmax(heatmap, dim):
    marginal = torch.sum(heatmap, dim=dim)
    sm = F.softmax(marginal, dim=2)
    return sm


def spacial_softmax(heatmap):
    height, width = heatmap.size(2), heatmap.size(3)
    h_sm, w_sm = marginal_softmax(heatmap, dim=3), marginal_softmax(heatmap, dim=2)
    hs = torch.linspace(0, 1, height).type_as(heatmap).expand(1, 1, -1).to(heatmap.device)
    ws = torch.linspace(0, 1, width).type_as(heatmap).expand(1, 1, -1).to(heatmap.device)
    h_k, w_k = torch.sum(h_sm * hs, dim=2, keepdim=True).squeeze(2), \
               torch.sum(w_sm * ws, dim=2, keepdim=True).squeeze(2)
    return torch.stack((h_k, w_k), dim=2)


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

