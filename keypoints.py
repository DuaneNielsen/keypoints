import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import vgg, keynet
from utils import ResultsLogger
from tps import tps_transform, tps_sample_params, rotate_affine_grid_multi
from apex import amp
from datasets import get_dataset
import argparse
import models.knn as knn


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='keypoint detection demo')

    """ config """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--run_id', type=int, required=True)
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--run_type', type=str, default='full')
    parser.add_argument('--train_mode', type=bool, default='True')
    parser.add_argument('--reload', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--opt_level', type=str, default='O0')

    """ visualization params"""
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--display_freq', type=int, default=10)
    parser.add_argument('--display_kp_rows', type=int, default=5)

    """ model parameters """
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--num_keypoints', type=int, default=10)

    """ hyper-parameters """
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)

    """ data and data augmentation parameters """
    parser.add_argument('--dataset', type=str, default='square')
    parser.add_argument('--tps_cntl_pts', type=int, default=4)
    parser.add_argument('--tps_variance', type=float, default=0.05)
    parser.add_argument('--max_rotate', type=float, default=0.1)

    args = parser.parse_args()

    """ logging """
    display = ResultsLogger(model_name=args.model_name,
                            run_id=args.run_id,
                            num_keypoints=args.num_keypoints,
                            title='Results',
                            visuals=args.display,
                            image_capture_freq=args.display_freq,
                            kp_rows=args.display_kp_rows,
                            comment=args.comment)
    display.header(args)

    """ dataset """
    train, test = get_dataset(args.data_root, args.dataset, args.run_type)
    pin_memory = False if args.device == 'cpu' else True
    train_l = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=pin_memory)
    test_l = DataLoader(test, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=pin_memory)

    """ data augmentation """

    def rand_peturb_params(batch_items):
        theta_tps, cntl_pts = tps_sample_params(batch_items, args.tps_cntl_pts, args.tps_variance)
        theta_rotate = torch.rand(batch_items) * 2 - 1
        theta_rotate = theta_rotate * args.max_rotate
        return theta_tps, cntl_pts, theta_rotate


    def peturb(x, tps_theta, cntl_pts, theta_rotate):
        x = tps_transform(x, tps_theta, cntl_pts)
        x = rotate_affine_grid_multi(x, theta_rotate)
        return x


    def augment(x):
        loss_mask = torch.ones(x.shape, dtype=x.dtype, device=x.device)
        bsize = x.size(0)
        theta_tps, cntl_pts, theta_rotate = rand_peturb_params(bsize)
        x = peturb(x, theta_tps, cntl_pts, theta_rotate)
        loss_mask = peturb(loss_mask, theta_tps, cntl_pts, theta_rotate)

        theta_tps, cntl_pts, theta_rotate = rand_peturb_params(bsize)
        x_ = peturb(x, theta_tps, cntl_pts, theta_rotate)
        loss_mask = peturb(loss_mask, theta_tps, cntl_pts, theta_rotate)
        return x, x_, loss_mask

    """ model """
    encoder_core = vgg.make_layers(vgg.vgg_cfg['F'])
    encoder = knn.Unit('encoder', 3, 64, encoder_core, out_batch_norm=False)
    decoder_core = vgg.make_layers(vgg.decoder_cfg['F'])
    decoder = knn.Unit('decoder', 64 + args.num_keypoints, 3, decoder_core)
    keypoint_core = vgg.make_layers(vgg.vgg_cfg['F'])
    keypoint = knn.Unit('keypoint', 3, args.num_keypoints, keypoint_core, out_batch_norm=False)
    keymapper = knn.GaussianLike(sigma=0.1)
    kp_network = keynet.KeyNet(args.model_name, encoder, keypoint, keymapper, decoder).to(args.device)

    if args.reload != 0:
        kp_network.load(args.reload)

    """ optimizer """
    if args.optimizer == 'Adam':
        optim = Adam(kp_network.parameters(), lr=1e-4)
    elif args.optimizer == 'SGD':
        optim = SGD(kp_network.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    scheduler = ReduceLROnPlateau(optim, mode='min')

    """ apex mixed precision """
    if args.device != 'cpu':
        model, optimizer = amp.initialize(kp_network, optim, opt_level=args.opt_level)

    """ loss function """
    def l2_reconstruction_loss(x, x_, loss_mask):
        loss = (x - x_) ** 2
        loss = loss * loss_mask
        return torch.mean(loss)

    criterion = l2_reconstruction_loss

    for epoch in range(1, args.epochs + 1):

        """ training """
        batch = tqdm(train_l, total=len(train) // args.batch_size)
        for i, (x, _) in enumerate(batch):
            x = x.to(args.device)
            x, x_, loss_mask = augment(x)

            optim.zero_grad()
            x_t, z, k, m, p, heatmap = kp_network(x, x_)

            loss = criterion(x_t, x_, loss_mask)

            if args.train_mode:
                if args.device != 'cpu':
                    with amp.scale_loss(loss, optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optim.step()

            display.log(batch, epoch, i, loss, optim, x, x_, x_t, heatmap, k, m, p, loss_mask, type='Train', depth=20)

        """ test  """
        with torch.no_grad():
            batch = tqdm(test_l, total=len(test) // args.batch_size)
            for i, (x, _) in enumerate(batch):
                x = x.to(args.device)
                x, x_, loss_mask = augment(x)

                x_t, z, k, m, p, heatmap = kp_network(x, x_)
                loss = criterion(x_t, x_, loss_mask)

                display.log(batch, epoch, i, loss, optim, x, x_, x_t, heatmap, k, m, p, loss_mask, type='Test', depth=20)

            ave_loss, best_loss = display.end_epoch(epoch, optim)
            scheduler.step(ave_loss)

            """ save if model improved """
            if ave_loss <= best_loss and args.train_mode:
                kp_network.save(args.run_id)
