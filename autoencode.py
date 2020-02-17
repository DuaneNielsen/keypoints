import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from colorama import Fore, Style
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import statistics as stats
from keypoints.models import vgg, knn, autoencoder
from utils import get_lr, UniImageViewer, make_grid
from keypoints.ds import datasets as ds
from apex import amp
from config import config

scale = 4
view_in = UniImageViewer('in', screen_resolution=(128 * 2 * scale, 128 * scale))
view_z = UniImageViewer('z', screen_resolution=(128//2 * 5 * scale, 128//2 * 4 * scale))


def log(phase):
    writer.add_scalar(f'{phase}_loss', loss.item(), global_step)

    if i % args.display_freq == 0:
        recon = torch.cat((x[0], x_[0]), dim=2)
        latent = make_grid(z[0].unsqueeze(1), 4, 4)
        if args.display:
            view_in.render(recon)
            view_z.render(latent)
        writer.add_image(f'{phase}_recon', recon, global_step)
        writer.add_image(f'{phase}_latent', latent.squeeze(0), global_step)


if __name__ == '__main__':

    args = config()
    torch.cuda.set_device(args.device)

    """ variables """
    best_loss = 100.0
    run_dir = f'data/models/{args.tag}/autoencode/{args.model_type}/run_{args.run_id}'
    writer = SummaryWriter(log_dir=run_dir)
    global_step = 0

    """ data """
    datapack = ds.datasets[args.dataset]
    train, test = datapack.make(args.dataset_train_len, args.dataset_test_len, data_root=args.data_root)
    train_l = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
    test_l = DataLoader(test, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)


    def add_co_ords_channels(x):
        """ adds 2 channels that carry co-ordinate information """
        b, h, w = x.size(0), x.size(2), x.size(3)
        hm = torch.linspace(0, 1, h, dtype=x.dtype, device=x.device).reshape(1, 1, h, 1).repeat(b, 1, 1, w)
        wm = torch.linspace(0, 1, w, dtype=x.dtype, device=x.device).reshape(1, 1, 1, w).repeat(b, 1, h, 1)
        return torch.cat((x, hm, wm), dim=1)

    """ model """
    nonlinearity, kwargs = nn.LeakyReLU, {"inplace": True}
    encoder_core = vgg.make_layers(vgg.vgg_cfg[args.model_type], nonlinearity=nonlinearity, nonlinearity_kwargs=kwargs)
    encoder = knn.Unit(args.model_in_channels, args.model_z_channels, encoder_core)
    decoder_core = vgg.make_layers(vgg.decoder_cfg[args.model_type], nonlinearity=nonlinearity, nonlinearity_kwargs=kwargs)
    decoder = knn.Unit(args.model_z_channels, args.model_in_channels, decoder_core)

    auto_encoder = autoencoder.AutoEncoder(encoder, decoder, init_weights=args.load is None).to(args.device)

    if args.load is not None:
        auto_encoder.load(args.load)

    """ optimizer """
    optim = Adam(auto_encoder.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optim, mode='min')

    """ apex mixed precision """
    if args.device != 'cpu':
        model, optimizer = amp.initialize(auto_encoder, optim, opt_level=args.opt_level)

    """ loss function """
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):

        """ training """
        batch = tqdm(train_l, total=len(train) // args.batch_size)
        for i, (x, _) in enumerate(batch):
            x = x.to(args.device)
            #x = add_co_ords_channels(x)

            optim.zero_grad()
            z, x_ = auto_encoder(x)
            loss = criterion(x_, x)
            if not args.demo:
                if args.device != 'cpu':
                    with amp.scale_loss(loss, optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optim.step()

            batch.set_description(f'Epoch: {epoch} LR: {get_lr(optim)} Train Loss: {loss.item()}')

            log('train')

            if i % args.checkpoint_freq == 0 and args.demo == 0:
                auto_encoder.save(run_dir + '/checkpoint')

            global_step += 1

        """ test  """
        with torch.no_grad():
            ll = []
            batch = tqdm(test_l, total=len(test) // args.batch_size)
            for i, (x, _) in enumerate(batch):
                x = x.to(args.device)

                z, x_ = auto_encoder(x)
                loss = criterion(x_, x)

                batch.set_description(f'Epoch: {epoch} Test Loss: {loss.item()}')
                ll.append(loss.item())
                log('test')

                global_step += 1

        """ check improvement """
        ave_loss = stats.mean(ll)
        scheduler.step(ave_loss)

        best_loss = ave_loss if ave_loss <= best_loss else best_loss
        print(f'{Fore.CYAN}ave loss: {ave_loss} {Fore.LIGHTBLUE_EX}best loss: {best_loss} {Style.RESET_ALL}')

        """ save if model improved """
        if ave_loss <= best_loss and not args.demo:
            auto_encoder.save(run_dir + '/best')


