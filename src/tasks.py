import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbMetricsLogger
from typing import Dict, Any
from keras.models import Model

from utils import get_optimizer
from model import build_xception_model, get_intermediate_layer_model
from dataset import load_dataset

def fit_model(dataset_dir, optimizer_type, learning_rate, epochs, steps_per_epoch, validation_steps, momentum=None):
    #TODO: pass arguments to the function
    model_name = 'xception'

    optimizer = get_optimizer(optimizer_type, learning_rate, momentum)

    model = build_xception_model()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    train_dir = dataset_dir + '/train'
    test_dir = dataset_dir + '/test'
    
    train_datagen = load_dataset(train_dir) # TODO: add preprocessing function and other parameters
    validation_datagen = load_dataset(test_dir) 

    wandb_config: Dict[str, Any] = {"model": {}}
    wandb_config['optimizer'] = {
        'type': optimizer_type,
        'learning_rate': learning_rate,
    }

    if momentum is not None:
        wandb_config['optimizer']['momentum'] = momentum

    wandb_config['epochs'] = epochs
    wandb_config['steps_per_epoch'] = steps_per_epoch
    wandb_config['validation_steps'] = validation_steps
    wandb_config['dataset_path'] = dataset_dir
    wandb.init(config=wandb_config, project='m3_week4', name=model_name)
    
    model.fit(
        train_datagen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_datagen,
        validation_steps=validation_steps,
        callbacks=[
            WandbMetricsLogger(),
            ]
    )
    wandb.finish()

    save_weights_dir = './model_weights/'
    save_plot_dir = './model_plots/'

    if not os.path.exists(save_weights_dir):
        os.makedirs(save_weights_dir)

    if not os.path.exists(save_plot_dir):
        os.makedirs(save_plot_dir)

    save_weights_path = save_weights_dir + model_name + '.h5'

    logging.info('Done!\n')
    logging.info('Saving the model into ' + save_weights_path + ' \n')

    model.save_weights(save_weights_path)

    logging.info('Done!\n')


def eval_model(model, test_generator):
	pass


def visualize_layer(model, sample, layer_index = -2, aggr = 'Max'):
    feature_model = get_intermediate_layer_model(model, layer_index)

    feature_maps = feature_model.predict(sample)
    feature_maps = feature_maps[0]

    plt.figure(figsize=(20,20))

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