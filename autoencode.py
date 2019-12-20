import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from colorama import Fore, Style
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import statistics as stats
from models import vgg, knn, autoencoder
from utils import get_lr, UniImageViewer
import datasets as ds
import argparse
from apex import amp

view_in = UniImageViewer('in', screen_resolution=(448, 192))

if __name__ == '__main__':

    """ config """
    parser = argparse.ArgumentParser(description='autoencoder for pre-training')

    """ config """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--run_id', type=int, required=True)
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--train_mode', type=bool, default='True')
    parser.add_argument('--reload', type=int, default=0)
    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--checkpoint_freq', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--opt_level', type=str, default='O0')

    """ visualization params"""
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--display_freq', type=int, default=10)
    parser.add_argument('--display_kp_rows', type=int, default=5)

    """ model parameters """
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--model_image_channels', type=int, default=3)

    """ hyper-parameters """
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)

    """ data and data augmentation parameters """
    parser.add_argument('--dataset', type=str, default='square')
    parser.add_argument('--dataset_size', type=str, default='full')

    args = parser.parse_args()

    """ variables """
    best_loss = 100.0

    """ data """
    train, test = ds.get_dataset(args.data_root, args.dataset, args.dataset_size)
    train_l = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
    test_l = DataLoader(test, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)

    """ model """
    nonlinearity, kwargs = nn.LeakyReLU, {"inplace": True}
    encoder_core = vgg.make_layers(vgg.vgg_cfg[args.model_type], nonlinearity=nonlinearity, nonlinearity_kwargs=kwargs)
    encoder = knn.Unit('encoder', args.model_image_channels, 64, encoder_core)
    decoder_core = vgg.make_layers(vgg.decoder_cfg[args.model_type], nonlinearity=nonlinearity, nonlinearity_kwargs=kwargs)
    decoder = knn.Unit('decoder', 64, args.model_image_channels, decoder_core)

    auto_encoder = autoencoder.AutoEncoder(args.model_name, encoder, decoder, init_weights=args.reload == 0).to(args.device)

    if args.reload != 0:
        auto_encoder.load(args.reload)

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
        ll = []
        batch = tqdm(train_l, total=len(train) // args.batch_size)
        for i, (x, _) in enumerate(batch):
            x = x.to(args.device)

            optim.zero_grad()
            z, x_ = auto_encoder(x)
            loss = criterion(x_, x)
            if args.train_mode:
                if args.device != 'cpu':
                    with amp.scale_loss(loss, optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optim.step()

            ll.append(loss.item())
            batch.set_description(f'Epoch: {epoch} LR: {get_lr(optim)} Train Loss: {stats.mean(ll)}')

            if not i % args.display_freq:
                view_in.render(torch.cat((x[0], x_[0]), dim=2))

            if i % args.checkpoint_freq == 0:
                auto_encoder.save(args.run_id, 'checkpoint')


        """ test  """
        with torch.no_grad():
            batch = tqdm(test_l, total=len(test) // args.batch_size)
            ll = []
            for i, (x, _) in enumerate(batch):
                x = x.to(args.device)

                z, x_ = auto_encoder(x)
                loss = criterion(x_, x)
                ll.append(loss.detach().item())
                batch.set_description(f'Epoch: {epoch} Test Loss: {stats.mean(ll)}')
                if not i % args.display_freq:
                    view_in.render(torch.cat((x[0], x_[0]), dim=2))

        """ check improvement """
        ave_loss = stats.mean(ll)
        scheduler.step(ave_loss)

        best_loss = ave_loss if ave_loss <= best_loss else best_loss
        print(f'{Fore.CYAN}ave loss: {ave_loss} {Fore.LIGHTBLUE_EX}best loss: {best_loss} {Style.RESET_ALL}')

        """ save if model improved """
        if ave_loss <= best_loss and args.train_mode:
            auto_encoder.save(args.run_id, 'best')


