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
from utils import get_lr, UniImageViewer, plot_keypoints_on_image
from tps import tps_random


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


view_in = UniImageViewer('in', screen_resolution=(128 * 3 * 4, 128 * 4))


def display(x, x_, k):
    height, width = x.size(2), x.size(3)
    key_x, key_y = k
    kp_image = plot_keypoints_on_image((key_x.float(), key_y.float()), x_.float())
    kp_image = transforms.Resize((height, width))(kp_image)
    kp_image_t = transforms.ToTensor()(kp_image).to(device)
    view_in.render(torch.cat((x[0].float(), x_[0].float(), x_t[0].float(), kp_image_t), dim=2))


if __name__ == '__main__':

    """ config """
    train_mode = True
    reload = False
    load_run_id = 4
    run_id = 5
    epochs = 800
    torchvision_data_root = 'data'
    model_name = 'vgg_kp_11'

    """ hyper-parameters"""
    batch_size = 96
    lr = 0.1

    """ variables """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_loss = 100.0
    max_i = 0

    """ data """
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
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
    kp_network = models.vgg11_bn_keypoint(sigma=0.1, num_keypoints=10, init_weights=True).to(device)

    kp_network.half()  # convert to half precision
    for layer in kp_network.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()

    if reload:
        load_model()

    """ optimizer """
    # optim = Adam(kp_network.parameters(), lr=1e-4)
    optim = SGD(kp_network.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optim, mode='min')

    """ loss function """
    criterion = nn.MSELoss()

    """ utils """

    for epoch in range(reload + 1, reload + epochs):

        ll = []
        """ training """
        batch = tqdm(train_l, total=len(train) // batch_size)
        for i, (x, _) in enumerate(batch):
            max_i = max_i if i < max_i else i
            x = x.half().to(device)
            x = tps_random(x, var=0.05)
            x_ = tps_random(x, var=0.05)

            optim.zero_grad()
            x_t, z, k = kp_network(x, x_)
            loss = criterion(x_t, x_)

            if train_mode:
                loss.backward()
                optim.step()

            ll.append(loss.item())
            if len(ll) > 20:
                ll.pop(0)
            batch.set_description(f'Epoch: {epoch} LR: {get_lr(optim)} Train Loss: {stats.mean(ll)}')

            if not i % 8:
                display(x, x_, k)

        """ test  """
        with torch.no_grad():
            batch = tqdm(test_l, total=len(test) // batch_size)
            del ll
            ll = []
            for i, (x, _) in enumerate(batch):
                x = x.half().to(device)
                x = tps_random(x, var=0.05)
                x_ = tps_random(x, var=0.05)

                x_t, z, k = kp_network(x, x_)
                loss = criterion(x_t, x_)
                ll.append(loss.detach().item())
                batch.set_description(f'Epoch: {epoch} Test Loss: {stats.mean(ll)}')
                if not i % 8:
                    display(x, x_, k)

        """ check improvement """
        ave_loss = stats.mean(ll)
        scheduler.step(ave_loss)

        best_loss = ave_loss if ave_loss <= best_loss else best_loss
        print(f'{Fore.CYAN}ave loss: {ave_loss} {Fore.LIGHTBLUE_EX}best loss: {best_loss} {Style.RESET_ALL}')

        """ save if model improved """
        if ave_loss <= best_loss and train_mode:
            save_model()
