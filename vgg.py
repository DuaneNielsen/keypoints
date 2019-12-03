import torch
import torchvision as tv
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from colorama import Fore, Style
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import statistics as stats

import models.classifier
from models import factory
from utils import precision, get_lr

if __name__ == '__main__':

    """ hyper-parameters"""
    batch_size = 512
    lr = 0.05

    num_classes = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_id = 2
    epochs = 200
    torchvision_data_root = '~/tv-data'

    best_precision = 0.0

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

    #classifier = vgg11_bn(num_classes=10, pretrained=False).to('cuda')
    classifier = factory.vgg11_bn(models.classifier.VGGClassifier(num_classes), pretrained=False).to('cuda')

    reload = False
    load_run_id = 2

    if reload:
        feature_block_load_path = Path('data/keypoints_vgg/models/run' + str(load_run_id) + '/feature.mdl')
        output_block_load_path = Path('data/keypoints_vgg/models/run' + str(load_run_id) + '/output.mdl')
        classifier.feature_block.load_state_dict((torch.load(str(feature_block_load_path))))
        classifier.output_block.load_state_dict(torch.load(str(output_block_load_path)))

    #optim = Adam(classifier.parameters(), lr=1e-3, weight_decay=5e-4)
    optim = SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optim, mode='max')
    criterion = nn.CrossEntropyLoss()

    for epoch in range(reload + 1, reload + epochs):

        ll = []
        batch = tqdm(train_l, total=len(train) // batch_size)
        for x, target in batch:
            x = x.to(device)
            target = target.to(device)

            optim.zero_grad()
            y = classifier(x)
            loss = criterion(y, target)
            loss.backward()
            optim.step()

            ll.append(loss.item())
            batch.set_description(f'Epoch: {epoch} LR: {get_lr(optim)} Train Loss: {stats.mean(ll)}')

        confusion = torch.zeros(num_classes, num_classes)
        batch = tqdm(test_l, total=len(test) // batch_size)
        ll = []
        for x, target in batch:
            x = x.to(device)
            target = target.to(device)

            y = classifier(x)
            loss = criterion(y, target)
            ll.append(loss.detach().item())
            batch.set_description(f'Epoch: {epoch} Test Loss: {stats.mean(ll)}')

            _, predicted = y.detach().max(1)

            for item in zip(predicted, target):
                confusion[item[0], item[1]] += 1

        precis, ave_precis = precision(confusion)

        print('')
        print(f'{Fore.CYAN}RESULTS FOR EPOCH {Fore.LIGHTYELLOW_EX}{epoch}{Style.RESET_ALL}')
        for i, cls in enumerate(classes):
            print(f'{Fore.LIGHTMAGENTA_EX}{cls} : {precis[i].item()}{Style.RESET_ALL}')
        best_precision = ave_precis if ave_precis > best_precision else best_precision
        print(f'{Fore.GREEN}ave precision : {ave_precis} best: {best_precision} {Style.RESET_ALL}')

        scheduler.step(ave_precis)

        if ave_precis >= best_precision:
            feature_block_save_path = Path('data/keypoints_vgg/models/run' + str(run_id) + '/feature.mdl')
            output_block_save_path = Path('data/keypoints_vgg/models/run' + str(run_id) + '/output.mdl')
            feature_block_save_path.parent.mkdir(parents=True, exist_ok=True)
            output_block_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(classifier.feature_block.state_dict(), str(feature_block_save_path))
            torch.save(classifier.output_block.state_dict(), str(output_block_save_path))
