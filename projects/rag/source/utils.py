import argparse

import yaml


def load_yaml(path: str):
    with open(path, 'r', encoding='UTF8') as f:
        load_yaml = yaml.load(f, Loader=yaml.FullLoader)
    return load_yaml


def get_argumnets():
    '''
    example
    python train.py --config_yaml base.yaml
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_yaml', type=str, default='config_yaml/base.yaml')

    args = parser.parse_args()

    return args
