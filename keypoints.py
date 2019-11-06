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
from tps import tps_random

view_in = UniImageViewer('in', screen_resolution=(224 * 3, 192))


class NANException(Exception):
    pass


if __name__ == '__main__':

    """ config """
    train_mode = True
    reload = True
    load_run_id = 3
    run_id = 4
    epochs = 800
    torchvision_data_root = 'data'
    model_name = 'vgg_kp_11'

    """ hyper-parameters"""
    batch_size = 32
    lr = 0.01

    """ variables """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_loss = 100.0
    kp_network = None
    max_i = 0

    while True:
        try:

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
            kp_network = models.vgg11_bn_keypoint(14, 12, sigma=0.1, num_keypoints=10, init_weights=True).to(device)
            #kp_network = models.vgg11_bn_keypoint_test(0, 0, num_keypoints=512).to(device)
            #kp_network = models.vgg11_bn_keypoint_test(0, 0, num_keypoints=0).to(device)

            if reload:
                encoder_block_load_path = Path(f'data/keypoint_net/{model_name}/run{str(load_run_id)}/encoder.mdl')
                decoder_block_load_path = Path(f'data/keypoint_net/{model_name}/run{str(load_run_id)}/decoder.mdl')
                keypoint_block_load_path = Path(f'data/keypoint_net/{model_name}/run{str(load_run_id)}/keypoint.mdl')
                kp_network.encoder.load_state_dict((torch.load(str(encoder_block_load_path))))
                kp_network.decoder.load_state_dict(torch.load(str(decoder_block_load_path)))
                kp_network.keypoint.load_state_dict(torch.load(str(keypoint_block_load_path)))

            """ optimizer """
            #optim = Adam(kp_network.parameters(), lr=1e-4)
            optim = SGD(kp_network.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            scheduler = ReduceLROnPlateau(optim, mode='min')

            """ loss function """
            criterion = nn.MSELoss()
            #criterion = models.PerceptualLoss().to(device)

            for epoch in range(reload + 1, reload + epochs):

                ll = []
                """ training """
                batch = tqdm(train_l, total=len(train) // batch_size)
                for i, (x, _) in enumerate(batch):
                    max_i = max_i if i < max_i else i
                    x = x.to(device)
                    x = tps_random(x, var=0.05)
                    x_ = tps_random(x, var=0.05)

                    optim.zero_grad()
                    x_t, z, k = kp_network(x, x_)
                    loss = criterion(x_t, x_)
                    if torch.isnan(loss):
                        raise NANException()

                    if train_mode:
                        loss.backward()
                        #print(k[0].max().item(), k[1].max().item())
                        #print(kp_network.keypoint.reducer._modules['0'].weight.grad)
                        optim.step()

                    ll.append(loss.item())
                    if len(ll) > 20:
                        ll.pop(0)
                    batch.set_description(f'Epoch: {epoch} LR: {get_lr(optim)} Train Loss: {stats.mean(ll)}')

                    if not i % 8:
                        #print(kp_network.decoder._modules['0'].weight.grad.data.norm().item())
                        view_in.render(torch.cat((x[0], x_[0], x_t[0]), dim=2))

                """ test  """
                with torch.no_grad():
                    batch = tqdm(test_l, total=len(test) // batch_size)
                    ll = []
                    for i, (x, _) in enumerate(batch):
                        x = x.to(device)
                        x = tps_random(x, var=0.05)
                        x_ = tps_random(x, var=0.05)

                        x_t, z, k = kp_network(x, x_)
                        loss = criterion(x_t, x_)
                        ll.append(loss.detach().item())
                        batch.set_description(f'Epoch: {epoch} Test Loss: {stats.mean(ll)}')
                        if not i % 8:
                            view_in.render(torch.cat((x[0], x_[0], x_t[0]), dim=2))

                """ check improvement """
                ave_loss = stats.mean(ll)
                scheduler.step(ave_loss)

                best_loss = ave_loss if ave_loss <= best_loss else best_loss
                print(f'{Fore.CYAN}ave loss: {ave_loss} {Fore.LIGHTBLUE_EX}best loss: {best_loss} {Style.RESET_ALL}')

                """ save if model improved """
                if ave_loss <= best_loss and train_mode:
                    encoder_block_save_path = Path(f'data/keypoint_net/{model_name}/run{str(run_id)}/encoder.mdl')
                    decoder_block_save_path = Path(f'data/keypoint_net/{model_name}/run{str(run_id)}/decoder.mdl')
                    keypoint_block_save_path = Path(f'data/keypoint_net/{model_name}/run{str(run_id)}/keypoint.mdl')
                    encoder_block_save_path.parent.mkdir(parents=True, exist_ok=True)
                    decoder_block_save_path.parent.mkdir(parents=True, exist_ok=True)
                    keypoint_block_save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(kp_network.encoder.state_dict(), str(encoder_block_save_path))
                    torch.save(kp_network.decoder.state_dict(), str(decoder_block_save_path))
                    torch.save(kp_network.keypoint.state_dict(), str(keypoint_block_save_path))
        except (NANException):
            print("restarting")
            print(max_i)
            print(best_loss)
            del kp_network
            del loss
            del x, x_, x_t, z, k, ll
            torch.cuda.empty_cache()
