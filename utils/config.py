import os
import yaml

def config_from_file():
    config_path = os.path.join(os.path.dirname(
                               os.path.dirname(
                                   os.path.abspath(__file__))), 'config', 'config.yaml')
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file)

    return config


if __name__ == '__main__':
    print(config_from_file())
