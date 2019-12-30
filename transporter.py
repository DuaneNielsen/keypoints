import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam

from data_augments import TpsAndRotate, nop
from models import vgg, transporter
from utils import ResultsLogger
from apex import amp
from datasets import get_dataset
from config import config
import models.knn as knn


if __name__ == '__main__':

    args = config()
    torch.cuda.set_device(args.device)
    run_dir = f'data/models/transporter/{args.model_type}/run_{args.run_id}'

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
    if args.data_aug_type == 'tps_and_rotate':
        augment = TpsAndRotate(args.data_aug_tps_cntl_pts, args.data_aug_tps_variance, args.data_aug_max_rotate)
    else:
        augment = nop

    """ model """
    nonlinearity, kwargs = nn.LeakyReLU, {"inplace": True}
    encoder_core = vgg.make_layers(vgg.vgg_cfg[args.model_type], nonlinearity=nonlinearity, nonlinearity_kwargs=kwargs)
    encoder = knn.Unit(args.model_in_channels, args.model_z_channels, encoder_core)
    decoder_core = vgg.make_layers(vgg.decoder_cfg[args.model_type])
    decoder = knn.Unit(args.model_z_channels, args.model_in_channels, decoder_core)
    keypoint_core = vgg.make_layers(vgg.vgg_cfg[args.model_type], nonlinearity=nonlinearity, nonlinearity_kwargs=kwargs)
    keypoint = knn.Unit(args.model_in_channels, args.model_keypoints, keypoint_core)
    keymapper = knn.GaussianLike(sigma=0.1)
    #mapper_core = vgg.make_layers(vgg.vgg_cfg['MAPPER'])
    #mapper_u = knn.Unit(args.model_keypoints, 1, mapper_core)
    #mapper = transporter.TransporterMap(mapper=mapper_u)
    #mapper.load('data/models/mapper/MAPPER/run_1/best')
    #keymapper = transporter.MaskMaker(mapper)
    for p in keymapper.parameters(recurse=True):
        p.requires_grad = False
    transporter_net = transporter.TransporterNet(encoder, keypoint, keymapper, decoder, init_weights=True,
                                                 combine_method=args.transporter_combine_mode)
    transporter_net = transporter_net.to(args.device)

    if args.load is not None:
        transporter_net.load(args.load)
    if args.transfer_load is not None:
        transporter_net.load_from_autoencoder(args.transfer_load)

    """ optimizer """
    optim = Adam(transporter_net.parameters(), lr=1e-4)

    """ apex mixed precision """
    if args.device != 'cpu':
        amp.initialize(transporter_net, optim, opt_level=args.opt_level)

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
                x_t, z, k, m, p, heatmap, mask_xs, mask_xt = transporter_net(x, x_)

                loss = criterion(x_t, x_, loss_mask)

                if args.device != 'cpu':
                    with amp.scale_loss(loss, optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optim.step()

                if i % args.checkpoint_freq == 0:
                    transporter_net.save(run_dir + '/checkpoint')

                display.log(batch, epoch, i, loss, optim, x, x_, x_t, heatmap, k, m, p, loss_mask,
                            type='train', depth=20, mask_xs=mask_xs, mask_xt=mask_xt)

        """ test  """
        with torch.no_grad():
            batch = tqdm(test_l, total=len(test) // args.batch_size)
            for i, data in enumerate(batch):
                data = to_device(data, device=args.device)
                x, x_, loss_mask = augment(*data)

                x_t, z, k, m, p, heatmap, mask_xs, mask_xt = transporter_net(x, x_)
                loss = criterion(x_t, x_, loss_mask)

                display.log(batch, epoch, i, loss, optim, x, x_, x_t, heatmap, k, m, p, loss_mask,
                            type='test', depth=20, mask_xs=mask_xs, mask_xt=mask_xt)

            ave_loss, best_loss = display.end_epoch(epoch, optim)

            """ save if model improved """
            if ave_loss <= best_loss and not args.demo:
                transporter_net.save(run_dir + '/best')
