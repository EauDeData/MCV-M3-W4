import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import wandb
import optuna
import datetime
import keras
from wandb.keras import WandbMetricsLogger
from typing import Dict, Any, List
from keras.models import Model

import utils
from model import build_xception_model, get_intermediate_layer_model
import dataset as dtst


def eval_model(model, test_generator) -> List[float]:
    # Evaluate the model on the test set
    logging.info('Evaluating the model on the test set...\n')
    scores = model.evaluate(test_generator)
    logging.info('Done!\n')

    # Print the accuracy and loss of the model on the test set
    logging.info('Test loss: {}'.format(scores[0]))
    logging.info('Test accuracy: {}\n'.format(scores[1]))

    return scores


def fit_model(model, train_set, val_set, config: Dict[str, Any], log2wandb: bool = True, save_weights: bool = True) -> List[float]:
    optimizer_config = config["optimizer"]
    optimizer = utils.get_optimizer(
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

def train_properly_implemented(model, train_set, val_set, optimizer_type, learning_rate, epochs, momentum=None):
    print('EPOCHES', epochs)
    model_name = 'xception'
    optimizer = utils.get_optimizer(optimizer_type, learning_rate, momentum)
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    wandb_config: Dict[str, Any] = {"model": {}}
    wandb_config['optimizer'] = {
        'type': optimizer_type,
        'learning_rate': learning_rate,
    }
    
    wandb_config['optimizer']['momentum'] = momentum
    wandb_config['epochs'] = epochs
    wandb.init(config=wandb_config, project='m3_week4', name=model_name)
    model.fit(train_set,
            epochs=epochs,
            validation_data=val_set,
            callbacks=[
                WandbMetricsLogger(),
                ])
    


def default_train(args, dataset_dir, experiment_name: str = None, log2wandb: bool = True, save_weights: bool = True) -> List[float]:
    ### DATA ###
    train_dir = dataset_dir + '/train'
    test_dir = dataset_dir + '/test'

    prep = keras.applications.xception.preprocess_input
    train_augmentations = utils.load_config(args.train_augmentations_file)

    train_datagen = dtst.load_dataset(
        train_dir, 
        target_size=args.image_size[0],
        batch_size=args.batch_size,
        preprocess_function=prep, 
        augmentations=train_augmentations
        )
    validation_datagen = dtst.load_dataset(
        test_dir,
        target_size=args.image_size[0],
        batch_size=args.batch_size,
        preprocess_function=prep
        )

    ### MODEL ###
    model = build_xception_model(args.model_weights_file)

    ### TRAIN LOOP ###
    if experiment_name is None:
        experiment_name = f'xception_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'

    config: Dict[str, Any] = {
        "experiment_name": experiment_name,
        "model": {
            "name": "Xception",
        },
        "optimizer": {
            'type': args.optimizer,
            'learning_rate': args.learning_rate,
        },
        "epochs": args.epochs,
        "dataset_path": args.dataset_dir,
    }

    if args.momentum is not None:
        config['optimizer']['momentum'] = args.momentum

    return fit_model(model, train_datagen, validation_datagen, config, log2wandb, save_weights)


def optuna_search(args, dataset_dir) -> List[float]:
    def objective(trial):
        train_dir = dataset_dir + '/train'
        test_dir = dataset_dir + '/test'

        prep = keras.applications.xception.preprocess_input

        train_augmentations = {
            'rotation_range': trial.suggest_int('rotation_range', 0, 180),
            'width_shift_range': trial.suggest_float('width_shift_range', 0.0, 1.0),
            'height_shift_range': trial.suggest_float('height_shift_range', 0.0, 1.0),
            'shear_range': trial.suggest_float('shear_range', 0.0, 1.0),
            'zoom_range': trial.suggest_float('zoom_range', 0.0, 1.0),
            'horizontal_flip': trial.suggest_categorical('horizontal_flip', [True, False]),
            'vertical_flip': trial.suggest_categorical('vertical_flip', [True, False]),
            'brightness_range': (trial.suggest_float('brightness_range_min', 0.0, 1.0), trial.suggest_float('brightness_range_max', 0.0, 1.0)),
            'fill_mode': trial.suggest_categorical('fill_mode', ['constant', 'nearest', 'reflect', 'wrap']),
            'cval': trial.suggest_float('cval', 0.0, 1.0),
        }

        train_set = dtst.load_dataset(
            train_dir,
            target_size=args.image_size[0],
            batch_size=args.batch_size,
            preprocess_function=prep,
            augmentations=train_augmentations
        )
        val_set = dtst.load_dataset(
            test_dir,
            target_size=args.image_size[0],
            batch_size=args.batch_size,
            preprocess_function=prep
        )

        config: Dict[str, Any] = {
            "experiment_name": f'xception_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}',
            "model": {
                "name": "Xception",
            },
            "optimizer": {
                'type': trial.suggest_categorical("optimizer", ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']),
                'learning_rate': trial.suggest_loguniform("learning_rate", 1e-5, 1e-1),
                'momentum': trial.suggest_float("momentum", 0.0, 1.0),
            },
            "epochs": args.epochs,
            "dataset_path": args.dataset_dir,
        }

        model = build_xception_model()
        return fit_model(model, train_set, val_set, config, log2wandb=False, save_weights=False)[1]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=224)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return study


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
