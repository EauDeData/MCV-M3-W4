import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbMetricsLogger
from typing import Dict, Any, List
from keras.models import Model

from utils import get_optimizer
from model import build_xception_model, get_intermediate_layer_model
from dataset import load_dataset


def eval_model(model, test_generator) -> List[float]:
    # Evaluate the model on the test set
    logging.info('Evaluating the model on the test set...\n')
    scores = model.evaluate(test_generator)
    logging.info('Done!\n')

    # Print the accuracy and loss of the model on the test set
    logging.info('Test loss: {}'.format(scores[0]))
    logging.info('Test accuracy: {}\n'.format(scores[1]))

    return scores


def fit_model(model, train_set, val_set, config: Dict[str, Any], log2wandb: bool = True) -> List[float]:
    optimizer_config = config["optimizer"]
    optimizer = get_optimizer(
        optimizer_config["type"],
        optimizer_config["learning_rate"],
        optimizer_config["momentum"]
    )

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    if log2wandb:
        wandb.init(config=config, project='m3_week4',
                name=config["experiment_name"])
        callbacks = [WandbMetricsLogger()]
    else:
        callbacks = []

    model.fit(
        train_set,
        epochs=config["epochs"],
        validation_data=val_set,
        callbacks=callbacks
    )

    if log2wandb:
        wandb.finish()

    save_weights_dir = './out/model_weights/'
    os.makedirs(save_weights_dir, exist_ok=True)

    save_weights_path = save_weights_dir + config["experiment_name"] + '.h5'

    logging.info('Done!\n')
    logging.info('Saving the model into ' + save_weights_path + ' \n')

    model.save_weights(save_weights_path)

    logging.info('Done!\n')

    # Return accuracy and loss of the model on the test set
    return eval_model(model, val_set)


def visualize_layer(model, sample, layer_index=-2, aggr='Max'):
    feature_model = get_intermediate_layer_model(model, layer_index)

    feature_maps = feature_model.predict(sample)
    feature_maps = feature_maps[0]

    plt.figure(figsize=(20, 20))

    if aggr == 'Max':
        layer_output = np.max(feature_maps, axis=-1)
    elif aggr == 'Avg':
        layer_output = np.average(feature_maps, axis=-1)
    elif aggr == 'Sum':
        layer_output = np.sum(feature_maps, axis=-1)
    else:
        raise ValueError('Invalid aggregation method')

    plt.imshow(layer_output)
    plt.savefig('feature_maps.png')


if __name__ == "__main__":
    """
    This is a sample script to show how to visualize the feature maps of a convolutional neural network.
    """
    from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
    from keras.models import Model
    import numpy as np

    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    flat = Flatten()(pool2)
    dense = Dense(10, activation='softmax')(flat)
    model = Model(input_layer, dense)

    sample = np.random.rand(1, 28, 28, 1)

    layer_index = 3

    # call the visualize_feature_maps function
    visualize_layer(model, sample, layer_index)
