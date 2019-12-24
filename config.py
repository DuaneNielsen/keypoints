import yaml
import argparse
from pathlib import Path
import torch


def config():
    """
    Reads the command switches and creates a config
    The --config file overrides command line switches
    :return:
    """

    """ config """
    parser = argparse.ArgumentParser(description='load_yaml_file')
    parser.add_argument('--device', type=str)
    parser.add_argument('--run_id', type=int, required=True)
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--transfer_load', type=str, default=None)
    parser.add_argument('--checkpoint_freq', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--opt_level', type=str, default='O0')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--tag', type=str, default='dev')

    """ visualization params """
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--display_freq', type=int, default=10)
    parser.add_argument('--display_kp_rows', type=int, default=5)

    """ model parameters """
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--model_image_channels', type=int, default=3)
    parser.add_argument('--num_keypoints', type=int, default=10)

    """ hyper-parameters """
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)

    """ data and data augmentation parameters """
    parser.add_argument('--dataset', type=str, default='square')
    parser.add_argument('--dataset_train_len', type=int, default=None)
    parser.add_argument('--dataset_test_len', type=int, default=None)
    parser.add_argument('--dataset_randomize', type=int, default=None)

    parser.add_argument('--tps_cntl_pts', type=int, default=4)
    parser.add_argument('--tps_variance', type=float, default=0.05)
    parser.add_argument('--max_rotate', type=float, default=0.1)

    args = parser.parse_args()

    if args.config is not None:
        with Path(args.config).open() as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
            vars(args).update(conf)

    if args.device is None:
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)

    return args
