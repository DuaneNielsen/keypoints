import yaml
import argparse
from pathlib import Path
import torch


def config(args=None):
    """
    Reads the command switches and creates a config
    Command line switches override config files
    :return:
    """

    """ config """
    parser = argparse.ArgumentParser(description='load_yaml_file')
    parser.add_argument('-d', '--device', type=str)
    parser.add_argument('-r', '--run_id', type=int, default=-1)
    parser.add_argument('--comment', type=str)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('-l', '--load', type=str, default=None)
    parser.add_argument('--transfer_load', type=str, default=None)
    parser.add_argument('--checkpoint_freq', type=int)
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('-c', '--config', type=str, default=None)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--processes', type=int)
    parser.add_argument('--seed', type=int, default=None)

    """cma params"""
    parser.add_argument('--cma_algo', type=str)
    parser.add_argument('--cma_step_mode', type=str)
    parser.add_argument('--cma_step_decay', type=float, default=None)
    parser.add_argument('--cma_initial_step_size', type=float, default=None)
    parser.add_argument('--cma_samples', type=int, default=None)
    parser.add_argument('--cma_oversample', type=int)

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

    """ policy parameters """
    parser.add_argument('--policy_action_select_mode', type=str)
    parser.add_argument('--policy_depth', type=int)

    """ gym env parameters """
    parser.add_argument('--gym_reward_count_limit', type=int, default=None)

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

    args = parser.parse_args(args)

    def set_if_not_set(args, dict):
        for key, value in dict.items():
            if key in vars(args) and vars(args)[key] is None:
                vars(args)[key] = dict[key]
            elif key not in vars(args):
                vars(args)[key] = dict[key]
        return args

    if args.config is not None:
        with Path(args.config).open() as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
            args = set_if_not_set(args, conf)

    defaults = {
        'checkpoint_freq': 1000,
        'opt_level': 'O0',
        'display_kp_rows': 4,
        'display_freq': 5000,

        'transporter_combine_mode': 'max',

        'processes': 7,
        'cma_algo;': 'fast',
        'cma_step_mode': 'auto',
        'cma_step_decay': 0.001,
        'cma_oversample': 0,

        'policy_action_select_mode': 'argmax',
        'policy_depth': 1
    }

    args = set_if_not_set(args, defaults)

    if args.device is None:
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)

    def counter():
        run_id_pid = Path('./.run_id')
        count = 1
        if run_id_pid.exists():
            with run_id_pid.open('r+') as f:
                last_id = int(f.readline())
                last_id += 1
                count = last_id
                f.seek(0)
                f.write(str(last_id))
        else:
            with run_id_pid.open('w+') as f:
                f.write(str(count))
        return count

    if args.run_id == -1:
        args.run_id = counter()

    return args
