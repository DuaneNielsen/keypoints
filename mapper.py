import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from colorama import Fore, Style
from torch.optim import Adam
import torch.nn as nn
import statistics as stats
from keypoints.models import vgg, knn, transporter
from utils import get_lr, UniImageViewer
from keypoints.ds import datasets as ds
from apex import amp
from config import config

scale = 4
view_in = UniImageViewer('in', screen_resolution=(128 * 2 * scale, 128 * scale))
view_z = UniImageViewer('z', screen_resolution=(128//2 * 5 * scale, 128//2 * 4 * scale))


def log(phase):
    writer.add_scalar(f'{phase}_loss', loss.item(), global_step)

    if i % args.display_freq == 0:
        recon = torch.cat((x_[0], mask[0]), dim=2)
        if args.display:
            view_in.render(recon)
            #view_z.render(latent)
        writer.add_image(f'{phase}_recon', recon, global_step)
        #writer.add_image(f'{phase}_latent', latent.squeeze(0), global_step)


if __name__ == '__main__':

    args = config()
    torch.cuda.set_device(args.device)

    """ variables """
    best_loss = 100.0
    run_dir = f'data/models/mapper/{args.model_type}/run_{args.run_id}'
    writer = SummaryWriter(log_dir=run_dir)
    global_step = 0

    """ data """
    dataset = ds.MapperDataset(32, 32, args.model_keypoints, args.dataset_train_len + args.dataset_test_len)
    train, test = ds.random_split(dataset, [args.dataset_train_len, args.dataset_test_len])
    train_l = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    test_l = DataLoader(test, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)

    """ model """
    encoder_core = vgg.make_layers(vgg.vgg_cfg[args.model_type])
    encoder = knn.Unit(args.model_keypoints, 1, encoder_core)
    mapper = transporter.TransporterMap(mapper=encoder, init_weights=args.load is None).to(args.device)

    if args.load is not None:
        mapper.load(args.load)

    """ optimizer """
    optim = Adam(mapper.parameters(), lr=1e-4)

    """ apex mixed precision """
    if args.device != 'cpu':
        model, optimizer = amp.initialize(mapper, optim, opt_level=args.opt_level)

    """ loss function """
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):

        """ training """
        batch = tqdm(train_l, total=len(train) // args.batch_size)
        for i, (pointmap, mask) in enumerate(batch):
            pointmap, mask = pointmap.to(args.device), mask.to(args.device)
            optim.zero_grad()
            x_ = mapper(pointmap)
            loss = criterion(x_, mask)
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
                mapper.save(run_dir + '/checkpoint')

            global_step += 1

        """ test  """
        with torch.no_grad():
            ll = []
            batch = tqdm(test_l, total=len(test) // args.batch_size)
            for i, (pointmap, mask) in enumerate(batch):
                pointmap, mask = pointmap.to(args.device), mask.to(args.device)
                x_ = mapper(pointmap)
                loss = criterion(x_, mask)

                batch.set_description(f'Epoch: {epoch} Test Loss: {loss.item()}')
                ll.append(loss.item())
                log('test')

                global_step += 1

        """ check improvement """
        ave_loss = stats.mean(ll)

        best_loss = ave_loss if ave_loss <= best_loss else best_loss
        print(f'{Fore.CYAN}ave loss: {ave_loss} {Fore.LIGHTBLUE_EX}best loss: {best_loss} {Style.RESET_ALL}')

        """ save if model improved """
        if ave_loss <= best_loss and not args.demo:
            mapper.save(run_dir + '/best')


