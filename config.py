import yaml
import argparse
from pathlib import Path
import torch


def config():
    """
    Reads the command switches and creates a config
    Command line switches override config files
    :return:
    """

    """ config """
    parser = argparse.ArgumentParser(description='load_yaml_file')
    parser.add_argument('-d', '--device', type=str)
    parser.add_argument('-r', '--run_id', type=int, required=True)
    parser.add_argument('--comment', type=str)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('-l', '--load', type=str, default=None)
    parser.add_argument('--transfer_load', type=str, default=None)
    parser.add_argument('--checkpoint_freq', type=int)
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--processes', type=int)

    """ visualization params """
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--display_freq', type=int)
    parser.add_argument('--display_kp_rows', type=int)

    """ model parameters """
    parser.add_argument('--opt_level', type=str)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--model_in_channels', type=int)
    parser.add_argument('--model_keypoints', type=int)
    parser.add_argument('--transporter_combine_mode', type=str)

    """ hyper-parameters """
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)

    """ data and data augmentation parameters """
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--dataset_train_len', type=int)
    parser.add_argument('--dataset_test_len', type=int)
    parser.add_argument('--dataset_randomize', type=int)

    parser.add_argument('--data_aug_tps_cntl_pts', type=int)
    parser.add_argument('--data_aug_tps_variance', type=float)
    parser.add_argument('--data_aug_max_rotate', type=float)

    args = parser.parse_args()

    def load_if_not_set(filepath, args):
        """
        Adds the update to args if it's not loaded
        :param filepath:
        :return:
        """
        with Path(filepath).open() as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
            for key, value in conf.items():
                if key in vars(args) and vars(args)[key] is None:
                    vars(args)[key] = conf[key]
                elif key not in vars(args):
                    vars(args)[key] = conf[key]
        return args

    if args.config is not None:
        args = load_if_not_set(args.config, args)

    args = load_if_not_set('configs/defaults.yaml', args)

    if args.device is None:
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)

    return args
