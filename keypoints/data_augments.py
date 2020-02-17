import torch

from tps import tps_sample_params, tps_transform, rotate_affine_grid_multi


def rand_peturb_params(batch_items, tps_cntl_pts, tps_variance, max_rotate):
    theta_tps, cntl_pts = tps_sample_params(batch_items, tps_cntl_pts, tps_variance)
    theta_rotate = torch.rand(batch_items) * 2 - 1
    theta_rotate = theta_rotate * max_rotate
    return theta_tps, cntl_pts, theta_rotate


def peturb(x, tps_theta, cntl_pts, theta_rotate):
    x = tps_transform(x, tps_theta, cntl_pts)
    x = rotate_affine_grid_multi(x, theta_rotate)
    return x


def nop(*data):
    return data[0], data[1], None


class TpsAndRotate(object):
    def __init__(self, data_aug_tps_cntl_pts, data_aug_tps_variance, data_aug_max_rotate):
        self.tps_cntl_pts, self.tps_variance, self.max_rotate = data_aug_tps_cntl_pts, data_aug_tps_variance, data_aug_max_rotate

    def __call__(self, *data):
        x = data[0]
        loss_mask = torch.ones(x.shape, dtype=x.dtype, device=x.device)
        bsize = x.size(0)
        theta_tps, cntl_pts, theta_rotate = rand_peturb_params(bsize, self.tps_cntl_pts, self.tps_variance, self.max_rotate)
        x = peturb(x, theta_tps, cntl_pts, theta_rotate)
        loss_mask = peturb(loss_mask, theta_tps, cntl_pts, theta_rotate)

        theta_tps, cntl_pts, theta_rotate = rand_peturb_params(bsize, self.tps_cntl_pts, self.tps_variance, self.max_rotate)
        x_ = peturb(x, theta_tps, cntl_pts, theta_rotate)
        loss_mask = peturb(loss_mask, theta_tps, cntl_pts, theta_rotate)
        return x, x_, loss_mask