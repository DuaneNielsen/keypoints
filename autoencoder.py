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


view_in = UniImageViewer('in', screen_resolution=(448, 192))

if __name__ == '__main__':

    """ config """
    train_mode = False
    reload = True
    load_run_id = 2
    run_id = 3
    epochs = 800
    torchvision_data_root = '~/tv-data'
    model_name = 'vgg_auto_16'

    """ hyper-parameters"""
    batch_size = 16
    lr = 0.05

    """ variables """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_loss = 100.0

    """ data """
    transform = transforms.Compose([
        transforms.Resize((224, 192)),
        transforms.ToTensor(),
    ])

    path = Path(torchvision_data_root + '/celeba-low')
    data = tv.datasets.ImageFolder(str(path), transform=transform)
    train = torch.utils.data.Subset(data, range(190000))
    test = torch.utils.data.Subset(data, range(190001, len(data)))

    train_l = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    test_l = DataLoader(test, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    """ model """
    auto_encoder = models.vgg11_bn_auto().to(device)

    if reload:
        encoder_block_load_path = Path(f'data/keypoints_auto/{model_name}/run{str(load_run_id)}/encoder.mdl')
        decoder_block_load_path = Path(f'data/keypoints_auto/{model_name}/run{str(load_run_id)}/decoder.mdl')
        auto_encoder.encoder.load_state_dict((torch.load(str(encoder_block_load_path))))
        auto_encoder.decoder.load_state_dict(torch.load(str(decoder_block_load_path)))

    """ optimizer """
    optim = SGD(auto_encoder.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optim, mode='min')

    """ loss function """
    criterion = nn.MSELoss()
    #criterion = models.PerceptualLoss().to(device)

    for epoch in range(reload + 1, reload + epochs):

        """ training """
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
                view_in.render(torch.cat((x[0], x_[0]), dim=2))

        """ test  """
        with torch.no_grad():
            batch = tqdm(test_l, total=len(test) // batch_size)
            ll = []
            for i, (x, _) in enumerate(batch):
                x = x.to(device)

                z, x_ = auto_encoder(x)
                loss = criterion(x_, x)
                ll.append(loss.detach().item())
                batch.set_description(f'Epoch: {epoch} Test Loss: {stats.mean(ll)}')
                if not i % 8:
                    view_in.render(torch.cat((x[0], x_[0]), dim=2))

        """ check improvement """
        ave_loss = stats.mean(ll)
        scheduler.step(ave_loss)

        best_loss = ave_loss if ave_loss <= best_loss else best_loss
        print(f'{Fore.CYAN}ave loss: {ave_loss} {Fore.LIGHTBLUE_EX}best loss: {best_loss} {Style.RESET_ALL}')

        """ save if model improved """
        if ave_loss <= best_loss and train_mode:
            encoder_block_save_path = Path(f'data/keypoints_auto/{model_name}/run{str(run_id)}/encoder.mdl')
            decoder_block_save_path = Path(f'data/keypoints_auto/{model_name}/run{str(run_id)}/decoder.mdl')
            encoder_block_save_path.parent.mkdir(parents=True, exist_ok=True)
            decoder_block_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(auto_encoder.encoder.state_dict(), str(encoder_block_save_path))
            torch.save(auto_encoder.decoder.state_dict(), str(decoder_block_save_path))
