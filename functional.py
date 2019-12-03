import torch


def squared_diff(h, height):
    hs = torch.linspace(0, 1, height, device=h.device).type_as(h).expand(h.shape[0], h.shape[1], height)
    hm = h.expand(height, -1, -1).permute(1, 2, 0)
    hm = ((hs - hm) ** 2)
    return hm


def gaussian_like_function(kp, height, width, sigma=0.1):
    h, w = kp
    hm = squared_diff(h, height)
    wm = squared_diff(w, width)
    hm = hm.expand(width, -1, -1, -1).permute(1, 2, 3, 0)
    wm = wm.expand(height, -1, -1, -1).permute(1, 2, 0, 3)
    gm = - (hm + wm + 1e-6).sqrt_() / (2 * sigma ** 2)
    gm = torch.exp(gm)
    return gm