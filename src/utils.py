import json
import numpy as np
from typing import Optional
from keras import backend as K
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Adamax, Nadam
import keras
import tensorflow as tf

def get_optimizer(optimizer: str, learning_rate: float, momentum: Optional[float]):
    if optimizer == 'sgd':
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    elif optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate, momentum=momentum)
    elif optimizer == 'adagrad':
        optimizer = Adagrad(learning_rate=learning_rate)
    elif optimizer == 'adadelta':
        optimizer = Adadelta(learning_rate=learning_rate)
    elif optimizer == 'adamax':
        optimizer = Adamax(learning_rate=learning_rate)
    elif optimizer == 'nadam':
        optimizer = Nadam(learning_rate=learning_rate)
    else:
        raise ValueError(f'Unknown optimizer: {optimizer}')

    return optimizer


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


def re_init(model):
    for l in model.layers:
        if hasattr(l,"kernel_initializer"):
            l.kernel.assign(l.kernel_initializer(tf.shape(l.kernel)))
        if hasattr(l,"bias_initializer"):
            l.bias.assign(l.bias_initializer(tf.shape(l.bias)))
        if hasattr(l,"recurrent_initializer"):
            l.recurrent_kernel.assign(l.recurrent_initializer(tf.shape(l.recurrent_kernel)))
    return model

def print_sparsity(model):
    total_w = 0
    total_0 = 0

    for i, w in enumerate(model.get_weights()):
        print(
            "{} -- Total:{}, Zeros: {:.2f}%".format(
                model.weights[i].name, w.size, np.sum(w == 0) / w.size * 100
            )
        )
        total_w += w.size
        total_0 += np.sum(w == 0)

    print(f"\nTotal weights: {total_w}. Pruned weights: {total_w - total_0}. Total sparsity: {total_0 / total_w * 100}%\n\n")