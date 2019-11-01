import torch
import torchvision as tv
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from colorama import Fore, Style
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.nn as nn
import statistics as stats
import models
from utils import get_lr, UniImageViewer

view_in = UniImageViewer('in', screen_resolution=(64*4, 128*4))

if __name__ == '__main__':

    train_mode = False
    reload = True
    load_run_id = 5

    """ hyper-parameters"""
    batch_size = 512
    lr = 1e-10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_id = 5
    epochs = 800
    torchvision_data_root = '~/tv-data'

    best_loss = 100.0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # image size 3, 32, 32
    # batch size must be an even number
    train = tv.datasets.CIFAR10(torchvision_data_root, train=True, download=True, transform=transform)
    test = tv.datasets.CIFAR10(torchvision_data_root, train=False, download=True, transform=transform)

    train_l = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_l = DataLoader(test, batch_size=batch_size, shuffle=True, drop_last=True)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    auto_encoder = models.vgg11_bn_auto().to(device)


    if reload:
        encoder_block_load_path = Path(f'data/keypoints_auto/models/run{str(load_run_id)}/encoder.mdl')
        decoder_block_load_path = Path(f'data/keypoints_auto/models/run{str(load_run_id)}/decoder.mdl')
        auto_encoder.encoder.load_state_dict((torch.load(str(encoder_block_load_path))))
        auto_encoder.decoder.load_state_dict(torch.load(str(decoder_block_load_path)))

    optim = SGD(auto_encoder.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optim, mode='min')
    criterion = nn.MSELoss()

    for epoch in range(reload + 1, reload + epochs):

        ll = []
        batch = tqdm(train_l, total=len(train) // batch_size)
        for i, (x, _) in enumerate(batch):
            x = x.to(device)

            optim.zero_grad()
            z, x_ = auto_encoder(x)
            loss = criterion(x_, x)
            if train_mode:
                loss.backward()
                optim.step()

            ll.append(loss.item())
            batch.set_description(f'Epoch: {epoch} LR: {get_lr(optim)} Train Loss: {stats.mean(ll)}')

            if not i % 8:
                view_in.render(torch.cat((x[0], x_[0]), dim=1))

        batch = tqdm(test_l, total=len(test) // batch_size)
        ll = []
        for i, (x, _) in enumerate(batch):
            x = x.to(device)

            z, x_ = auto_encoder(x)
            loss = criterion(x_, x)
            ll.append(loss.detach().item())
            batch.set_description(f'Epoch: {epoch} Test Loss: {stats.mean(ll)}')
            if not i % 8:
                if not i % 8:
                    view_in.render(torch.cat((x[0], x_[0]), dim=1))

        ave_loss = stats.mean(ll)
        scheduler.step(ave_loss)

        best_loss = ave_loss if ave_loss <= best_loss else best_loss
        print(f'{Fore.CYAN}ave loss: {ave_loss} {Fore.LIGHTBLUE_EX}best loss: {best_loss} {Style.RESET_ALL}')

        if ave_loss <= best_loss and train_mode:
            encoder_block_save_path = Path(f'data/keypoints_auto/models/run{str(run_id)}/encoder.mdl')
            decoder_block_save_path = Path(f'data/keypoints_auto/models/run{str(run_id)}/decoder.mdl')
            encoder_block_save_path.parent.mkdir(parents=True, exist_ok=True)
            decoder_block_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(auto_encoder.encoder.state_dict(), str(encoder_block_save_path))
            torch.save(auto_encoder.decoder.state_dict(), str(decoder_block_save_path))
