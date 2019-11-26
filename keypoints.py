import torch
import torch.nn as nn
import torchvision as tv
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from colorama import Fore, Style
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import statistics as stats
import models
from utils import get_lr, ResultsLogger
from tps import RandomTPSTransform, RandRotate
from apex import amp
import logging
from benchmark import SquareDataset
import argparse


def load_model(model_name, load_run_id):
    encoder_block_load_path = Path(f'data/keypoint_net/{model_name}/run{str(load_run_id)}/encoder.mdl')
    decoder_block_load_path = Path(f'data/keypoint_net/{model_name}/run{str(load_run_id)}/decoder.mdl')
    keypoint_block_load_path = Path(f'data/keypoint_net/{model_name}/run{str(load_run_id)}/keypoint.mdl')
    kp_network.encoder.load_state_dict((torch.load(str(encoder_block_load_path))))
    kp_network.decoder.load_state_dict(torch.load(str(decoder_block_load_path)))
    kp_network.keypoint.load_state_dict(torch.load(str(keypoint_block_load_path)))


def save_model(model_name, run_id):
    encoder_block_save_path = Path(f'data/keypoint_net/{model_name}/run{str(run_id)}/encoder.mdl')
    decoder_block_save_path = Path(f'data/keypoint_net/{model_name}/run{str(run_id)}/decoder.mdl')
    keypoint_block_save_path = Path(f'data/keypoint_net/{model_name}/run{str(run_id)}/keypoint.mdl')
    encoder_block_save_path.parent.mkdir(parents=True, exist_ok=True)
    decoder_block_save_path.parent.mkdir(parents=True, exist_ok=True)
    keypoint_block_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(kp_network.encoder.state_dict(), str(encoder_block_save_path))
    torch.save(kp_network.decoder.state_dict(), str(decoder_block_save_path))
    torch.save(kp_network.keypoint.state_dict(), str(keypoint_block_save_path))


class ConfigException(Exception):
    pass


def get_dataset(dataset, run_type):

    if dataset is '/celeba-low':
        path = Path(args.data_root + dataset)
        """ celeba a transforms """
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        data = tv.datasets.ImageFolder(str(path), transform=transform)
    elif dataset is 'square':
        data = SquareDataset(size=200000, transform=transforms.ToTensor())
    else:
        raise ConfigException('pick a dataset')

    if run_type is 'full':
        train = torch.utils.data.Subset(data, range(190000))
        test = torch.utils.data.Subset(data, range(190001, len(data)))
    elif run_type is 'small':
        train = torch.utils.data.Subset(data, range(10000))
        test = torch.utils.data.Subset(data, range(10001, 11001))
    elif run_type is 'short':
        train = torch.utils.data.Subset(data, range(2000))
        test = torch.utils.data.Subset(data, range(2001, 2501))
    else:
        raise ConfigException('pick a run type')

    return train, test


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='keypoint detection demo')

    parser.add_argument('--run_id', type=int, required=True)
    parser.add_argument('--run_type', type=str, default='short')
    parser.add_argument('--train_mode', type=bool, default='True')
    parser.add_argument('--reload', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--model_name', type=str, default='vgg_kp_11')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--opt_level', type=str, default='O0')
    parser.add_argument('--dataset', type=str, default='square')
    parser.add_argument('--num_keypoints', type=int, default=4)

    """ hyper-parameters """
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)

    args = parser.parse_args()

    """ variables """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    display = ResultsLogger(title='Results')
    display.header(args.run_id, args.model_name, args.run_type, args.batch_size, args.lr, args.opt_level, args.dataset, args.num_keypoints)

    """ dataset """
    train, test = get_dataset(args.dataset, args.run_type)
    train_l = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
    test_l = DataLoader(test, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)

    """ data augmentation"""
    peturb = transforms.Compose([
        RandRotate(),
        RandomTPSTransform()
    ])

    """ model """
    kp_network = models.vgg11_bn_keypoint(sigma=0.1, num_keypoints=args.num_keypoints, init_weights=True).to(device)

    if args.reload != 0:
        load_model(args.model_name, args.reload)

    """ optimizer """
    # optim = Adam(kp_network.parameters(), lr=1e-4)
    optim = SGD(kp_network.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optim, mode='min')

    """ apex """
    model, optimizer = amp.initialize(kp_network, optim, opt_level=args.opt_level)

    """ loss function """
    criterion = nn.MSELoss()
    #criterion = models.DiscountBlackLoss()

    for epoch in range(1, args.epochs):

        ll = []
        """ training """
        batch = tqdm(train_l, total=len(train) // args.batch_size)
        for i, (x, _) in enumerate(batch):
            x = x.to(device)
            x = peturb(x)
            x_ = peturb(x)

            optim.zero_grad()
            x_t, z, k = kp_network(x, x_)
            loss = criterion(x_t, x_)
            #loss, loss_image, loss_mask = criterion(x_t, x_)

            if args.train_mode:
                with amp.scale_loss(loss, optim) as scaled_loss:
                    scaled_loss.backward()
                optim.step()

            display.log(batch, epoch, i, loss, optim, x, x_, x_t, k, type='Train')

        """ test  """
        with torch.no_grad():
            batch = tqdm(test_l, total=len(test) // args.batch_size)
            ll = []
            for i, (x, _) in enumerate(batch):
                x = x.to(device)
                x = peturb(x)
                x_ = peturb(x)

                x_t, z, k = kp_network(x, x_)
                loss = criterion(x_t, x_)

                display.log(batch, epoch, i, loss, optim, x, x_, x_t, k, type='Test')

            ave_loss, best_loss = display.end_epoch(epoch, optim)
            scheduler.step(ave_loss)

            """ save if model improved """
            if ave_loss <= best_loss and args.train_mode:
                save_model(args.model_name, args.run_id)
