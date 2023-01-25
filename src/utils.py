import json
from typing import Optional
from keras.optimizers import SGD, Adam


def get_optimizer(optimizer: str, learning_rate: float, momentum: Optional[float]):
    if optimizer == 'sgd':
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    elif optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        raise Exception('Optimizer not supported: {}'.format(optimizer))

    return optimizer


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        config = json.load(f)

    return config