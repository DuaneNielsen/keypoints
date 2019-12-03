import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import factory
from utils import ResultsLogger
from tps import RandomTPSTransform, RandRotate
from apex import amp
from datasets import get_dataset
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='keypoint detection demo')

    """ config """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--run_id', type=int, required=True)
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--run_type', type=str, default='short')
    parser.add_argument('--train_mode', type=bool, default='True')
    parser.add_argument('--reload', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--opt_level', type=str, default='O0')
    parser.add_argument('--dataset', type=str, default='square')
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--optimizer', type=str, default='Adam')

    """ model parameters """
    parser.add_argument('--model_name', type=str, default='vgg_kp_11')
    parser.add_argument('--num_keypoints', type=int, default=4)

    """ hyper-parameters """
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)

    """ data augmentation parameters """
    parser.add_argument('--tps_cntl_pts', type=int, default=4)
    parser.add_argument('--tps_variance', type=float, default=0.11)
    parser.add_argument('--max_rotate', type=float, default=0.2)

    args = parser.parse_args()

    """ variables """
    display = ResultsLogger(model_name=args.model_name, run_id=args.run_id, title='Results', visuals=args.display, comment=args.comment)
    display.header(args)

    """ dataset """
    train, test = get_dataset(args.data_root, args.dataset, args.run_type)
    train_l = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
    test_l = DataLoader(test, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)

    """ data augmentation"""
    peturb = transforms.Compose([
        RandRotate(max=args.max_rotate),
        RandomTPSTransform(variance=args.tps_variance)
    ])

    """ model """
    kp_network = factory.vgg11_bn_keypoint(sigma=0.1, num_keypoints=args.num_keypoints, init_weights=True).to(args.device)

    if args.reload != 0:
        load_model(args.model_name, args.reload)

    """ optimizer """
    if args.optimizer == 'Adam':
        optim = Adam(kp_network.parameters(), lr=1e-4)
    elif args.optimizer == 'SGD':
        optim = SGD(kp_network.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    scheduler = ReduceLROnPlateau(optim, mode='min')

    """ apex """
    if args.device != 'cpu':
        model, optimizer = amp.initialize(kp_network, optim, opt_level=args.opt_level)

    """ loss function """
    criterion = nn.MSELoss()
    #criterion = models.DiscountBlackLoss()

    for epoch in range(1, args.epochs + 1):

        ll = []
        """ training """
        batch = tqdm(train_l, total=len(train) // args.batch_size)
        for i, (x, _) in enumerate(batch):
            x = x.to(args.device)
            x = peturb(x)
            x_ = peturb(x)

            optim.zero_grad()
            x_t, z, k = kp_network(x, x_)
            loss = criterion(x_t, x_)
            #loss, loss_image, loss_mask = criterion(x_t, x_)

            if args.train_mode:
                if args.device != 'cpu':
                    with amp.scale_loss(loss, optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optim.step()

            display.log(batch, epoch, i, loss, optim, x, x_, x_t, k, type='Train')

        """ test  """
        with torch.no_grad():
            batch = tqdm(test_l, total=len(test) // args.batch_size)
            ll = []
            for i, (x, _) in enumerate(batch):
                x = x.to(args.device)
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
