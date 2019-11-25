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


def load_model():
    encoder_block_load_path = Path(f'data/keypoint_net/{model_name}/run{str(load_run_id)}/encoder.mdl')
    decoder_block_load_path = Path(f'data/keypoint_net/{model_name}/run{str(load_run_id)}/decoder.mdl')
    keypoint_block_load_path = Path(f'data/keypoint_net/{model_name}/run{str(load_run_id)}/keypoint.mdl')
    kp_network.encoder.load_state_dict((torch.load(str(encoder_block_load_path))))
    kp_network.decoder.load_state_dict(torch.load(str(decoder_block_load_path)))
    kp_network.keypoint.load_state_dict(torch.load(str(keypoint_block_load_path)))


def save_model():
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
        path = Path(torchvision_data_root + dataset)
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

    """ config """
    #  run type = 'full' | 'small' | 'short'
    run_type = 'short'
    train_mode = True
    reload = True
    load_run_id = 11
    run_id = 12
    epochs = 800
    torchvision_data_root = 'data'
    model_name = 'vgg_kp_11'
    # Apex Mixed precision Initialization
    opt_level = 'O0'
    # dataset = '/celeba-low' | 'square'
    #dataset = '/celeba-low'
    dataset = 'square'
    num_keypoints = 4

    """ hyper-parameters """
    batch_size = 32
    lr = 0.001

    """ variables """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    display = ResultsLogger(title='Results')
    display.header(run_id, model_name, run_type, batch_size, lr, opt_level, dataset, num_keypoints)

    """ dataset """
    train, test = get_dataset(dataset, run_type)
    train_l = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    test_l = DataLoader(test, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

    """ data augmentation"""
    peturb = transforms.Compose([
        RandRotate(),
        RandomTPSTransform()
    ])

    """ model """
    kp_network = models.vgg11_bn_keypoint(sigma=0.1, num_keypoints=num_keypoints, init_weights=True).to(device)

    if reload:
        load_model()

    """ optimizer """
    # optim = Adam(kp_network.parameters(), lr=1e-4)
    optim = SGD(kp_network.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optim, mode='min')

    """ apex """
    model, optimizer = amp.initialize(kp_network, optim, opt_level=opt_level)

    """ loss function """
    criterion = nn.MSELoss()
    #criterion = models.DiscountBlackLoss()

    """ utils """

    for epoch in range(1, epochs):

        ll = []
        """ training """
        batch = tqdm(train_l, total=len(train) // batch_size)
        for i, (x, _) in enumerate(batch):
            x = x.to(device)
            x = peturb(x)
            x_ = peturb(x)

            optim.zero_grad()
            x_t, z, k = kp_network(x, x_)
            loss = criterion(x_t, x_)
            #loss, loss_image, loss_mask = criterion(x_t, x_)

            if train_mode:
                with amp.scale_loss(loss, optim) as scaled_loss:
                    scaled_loss.backward()
                optim.step()

            display.log(batch, epoch, i, loss, optim, x, x_, x_t, k, type='Train')

        """ test  """
        with torch.no_grad():
            batch = tqdm(test_l, total=len(test) // batch_size)
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
            if ave_loss <= best_loss and train_mode:
                save_model()
