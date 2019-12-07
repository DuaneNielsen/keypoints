import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from colorama import Fore, Style
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import statistics as stats
from models import vgg, knn, autoencoder
from utils import get_lr, UniImageViewer
import datasets as ds

view_in = UniImageViewer('in', screen_resolution=(448, 192))

if __name__ == '__main__':

    """ config """
    train_mode = True
    reload = False
    load_run_id = 3
    run_id = 3
    epochs = 800
    data_root = 'data'
    model_name = 'vgg_autoencoder_11'

    """ hyper-parameters"""
    batch_size = 16
    lr = 0.05

    """ variables """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_loss = 100.0

    """ data """
    train, test = ds.get_dataset(data_root, ds.D_CELEBA, ds.SIZE_SHORT, )
    train_l = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    test_l = DataLoader(test, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

    """ model """
    encoder_core = vgg.make_layers(vgg.vgg_cfg['F'])
    encoder = knn.Unit('encoder', 3, 64, encoder_core)
    decoder_core = vgg.make_layers(vgg.decoder_cfg['F'])
    decoder = knn.Unit('decoder', 64, 3, decoder_core)

    auto_encoder = autoencoder.AutoEncoder(model_name, encoder, decoder, init_weights=not reload).to(device)

    if reload:
        auto_encoder.load(load_run_id)

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
            auto_encoder.save(run_id)


