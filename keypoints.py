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
from config import config
import models.knn as knn


if __name__ == '__main__':

    args = config()
    torch.cuda.set_device(args.device)
    run_dir = f'data/models/{args.tag}/keypoints/{args.model_type}/run_{args.run_id}'

    """ logging """
    display = ResultsLogger(run_dir=run_dir,
                            num_keypoints=args.model_keypoints,
                            title='Results',
                            visuals=args.display,
                            image_capture_freq=args.display_freq,
                            kp_rows=args.display_kp_rows,
                            comment=args.comment)
    display.header(args)

    """ dataset """
    train, test = get_dataset(args.data_root, args.dataset,
                              args.dataset_train_len, args.dataset_test_len, args.dataset_randomize)
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

    def tps_and_rotate(*data):
        x = data[0]
        loss_mask = torch.ones(x.shape, dtype=x.dtype, device=x.device)
        bsize = x.size(0)
        theta_tps, cntl_pts, theta_rotate = rand_peturb_params(bsize)
        x = peturb(x, theta_tps, cntl_pts, theta_rotate)
        loss_mask = peturb(loss_mask, theta_tps, cntl_pts, theta_rotate)

        theta_tps, cntl_pts, theta_rotate = rand_peturb_params(bsize)
        x_ = peturb(x, theta_tps, cntl_pts, theta_rotate)
        loss_mask = peturb(loss_mask, theta_tps, cntl_pts, theta_rotate)
        return x, x_, loss_mask

    def nop(*data):
        return data[0], data[1], None

    if args.data_aug_type == 'tps_and_rotate':
        augment = tps_and_rotate
    else:
        augment = nop

    """ model """
    nonlinearity, kwargs = nn.LeakyReLU, {"inplace": True}
    encoder_core = vgg.make_layers(vgg.vgg_cfg[args.model_type], nonlinearity=nonlinearity, nonlinearity_kwargs=kwargs)
    encoder = knn.Unit(args.model_in_channels, args.model_z_channels, encoder_core)
    decoder_core = vgg.make_layers(vgg.decoder_cfg[args.model_type])
    decoder = knn.Unit(args.model_z_channels + args.model_keypoints, args.model_in_channels, decoder_core)
    keypoint_core = vgg.make_layers(vgg.vgg_cfg[args.model_type], nonlinearity=nonlinearity, nonlinearity_kwargs=kwargs)
    keypoint = knn.Unit(args.model_in_channels, args.model_keypoints, keypoint_core)
    keymapper = knn.GaussianLike(sigma=0.1)
    kp_network = keynet.KeyNet(encoder, keypoint, keymapper, decoder, init_weights=True)
    kp_network = kp_network.to(args.device)

    if args.load is not None:
        kp_network.load(args.load)
    if args.transfer_load is not None:
        kp_network.load_from_autoencoder(args.transfer_load)

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
    def l2_reconstruction_loss(x, x_, loss_mask=None):
        loss = (x - x_) ** 2
        if loss_mask is not None:
            loss = loss * loss_mask
        return torch.mean(loss)

    criterion = l2_reconstruction_loss

    def to_device(data, device):
        return tuple([x.to(device) for x in data])

    for epoch in range(1, args.epochs + 1):

        if not args.demo:
            """ training """
            batch = tqdm(train_l, total=len(train) // args.batch_size)
            for i, data in enumerate(batch):
                data = to_device(data, device=args.device)
                x, x_, loss_mask = augment(*data)

                optim.zero_grad()
                x_t, z, k, m, p, heatmap = kp_network(x, x_)

                loss = criterion(x_t, x_, loss_mask)

                if args.device != 'cpu':
                    with amp.scale_loss(loss, optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optim.step()

                if i % args.checkpoint_freq == 0:
                    kp_network.save(run_dir + '/checkpoint')

                display.log(batch, epoch, i, loss, optim, x, x_, x_t, heatmap, k, m, p, loss_mask, type='Train', depth=20)

        """ test  """
        with torch.no_grad():
            batch = tqdm(test_l, total=len(test) // args.batch_size)
            for i, data in enumerate(batch):
                data = to_device(data, device=args.device)
                x, x_, loss_mask = augment(*data)

                x_t, z, k, m, p, heatmap = kp_network(x, x_)
                loss = criterion(x_t, x_, loss_mask)

                display.log(batch, epoch, i, loss, optim, x, x_, x_t, heatmap, k, m, p, loss_mask, type='Test', depth=20)

            ave_loss, best_loss = display.end_epoch(epoch, optim)
            scheduler.step(ave_loss)

            """ save if model improved """
            if ave_loss <= best_loss and not args.demo:
                kp_network.save(run_dir + '/best')
