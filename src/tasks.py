import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import wandb
import optuna
import datetime
import keras
import pickle
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from wandb.keras import WandbMetricsLogger
from typing import Dict, Any, List
from keras.models import Model

import utils
from model import build_xception_model, build_model_tricks, get_baseline_cnn, get_intermediate_layer_model

import dataset as dtst
import distilator


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

    if save_weights:
        logging.info('Saving the model into ' + save_weights_path + ' \n')
        model.save_weights(save_weights_path)
        logging.info('Done!\n')

    # Return accuracy and loss of the model on the test set
    return eval_model(model, val_set)


def train_properly_implemented(
    model, train_set, val_set, optimizer_type, learning_rate, epochs, momentum=None, save_weights=True
):
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

    save_weights_dir = './out/model_weights/'
    os.makedirs(save_weights_dir, exist_ok=True)

    save_weights_path = save_weights_dir + 'xceptionSmallData' + '.h5'

    logging.info('Done!\n')

    if save_weights:
        logging.info('Saving the model into ' + save_weights_path + ' \n')
        model.save_weights(save_weights_path)
        logging.info('Done!\n')


def train_tricks_train(model, train_set, val_set, optimizer_type, learning_rate, epochs, reduce = 0.1, momentum=None):
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
    scheduler = keras.callbacks.ReduceLROnPlateau(monitor = "loss", min_delta = 0.001, factor = reduce, patience = 5)
    print('TRAINING WITH SCHEDULER')
    model.fit(train_set,
            epochs=epochs,
            validation_data=val_set,
            callbacks=[scheduler,
                WandbMetricsLogger(),
                ])

def default_train(args, dataset_dir, experiment_name: str = None, log2wandb: bool = True, save_weights: bool = True) -> List[float]:
    ### DATA ###
    train_dir = dataset_dir + '/train'
    test_dir = dataset_dir + '/test'

    prep = keras.applications.xception.preprocess_input

    if args.train_augmentations_file is not None:
        train_augmentations = utils.load_config(args.train_augmentations_file)
    else:
        train_augmentations = {}

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


def optuna_search(args, dataset_dir, n_trials=200) -> List[float]:
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

        freeze_from = trial.suggest_float('freeze_from', 0, 0.9)
        freeze_percent = trial.suggest_float('freeze_percent', 0.1, 1.0 - freeze_from)

        config: Dict[str, Any] = {
            "experiment_name": f'xception_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}',
            "model": {
                "name": "Xception",
                "dropout": trial.suggest_categorical("dropout", [True, False]),
                "batch_norm": trial.suggest_categorical("batch_norm", [True, False]),
                "regularizer": trial.suggest_categorical("regularizer", [True, False]),
                "freeze_from": freeze_from,
                "freeze_percent": freeze_percent,
            },
            "optimizer": {
                'type': trial.suggest_categorical("optimizer", ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']),
                'learning_rate': trial.suggest_loguniform("learning_rate", 1e-5, 1e-1),
                'momentum': trial.suggest_float("momentum", 0.0, 1.0),
            },
            "epochs": args.epochs,
            "dataset_path": args.dataset_dir,
        }

        model = build_model_tricks(
            dropout=config['model']['dropout'],
            batch_norm=config['model']['batch_norm'],
            regularizer=config['model']['regularizer'],
            freeze_from=config['model']['freeze_from'],
            freeze_percent=config['model']['freeze_percent'],
        )
        return fit_model(model, train_set, val_set, config, log2wandb=False, save_weights=False)[1]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save report with accuracy and best params to file
    with open(f'report_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.txt', 'w') as f:
        f.write(f'Best accuracy: {trial.value}\n\n')

        f.write('Best params: \n')
        for key, value in trial.params.items():
            f.write(f'{key}: {value}\n')

    # Save study object to file
    with open(f'study_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.pkl', 'wb') as f:
        pickle.dump(study, f)

    return study


def build_optuna_model(args, report_file: str = "report_best_model.txt"):
    # Load dataset
    dataset_dir = args.dataset_dir
    train_dir = dataset_dir + '/train'
    test_dir = dataset_dir + '/test'

    # Load report with best params
    with open(report_file, 'r') as f:
        lines = f.readlines()

    # Get best params
    best_params = {}
    for line in lines[3:]:
        key, value = line.split(': ')
        best_params[key] = value.strip()

    augmentations = {
        'rotation_range': float(best_params['rotation_range']),
        'width_shift_range': float(best_params['width_shift_range']),
        'height_shift_range': float(best_params['height_shift_range']),
        'shear_range': float(best_params['shear_range']),
        'zoom_range': float(best_params['zoom_range']),
        'horizontal_flip': best_params['horizontal_flip'] == 'True',
        'vertical_flip': best_params['vertical_flip'] == 'True',
        'brightness_range': (float(best_params['brightness_range_min']), float(best_params['brightness_range_max'])),
        'fill_mode': best_params['fill_mode'],
        'cval': float(best_params['cval']),
    }

    config = {
        "experiment_name": f'xception_best_model_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}',
        "model": {
            "name": "Xception",
            "dropout": best_params['dropout'] == 'True',
            "batch_norm": best_params['batch_norm'] == 'True',
            "regularizer": best_params['regularizer'] == 'True',
            "freeze_from": float(best_params['freeze_from']),
            "freeze_percent": float(best_params['freeze_percent']),
        },
        "optimizer": {
            'type': best_params['optimizer'],
            'learning_rate': float(best_params['learning_rate']),
            'momentum': float(best_params['momentum']),
        },
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "dataset_path": args.dataset_dir,
    }

    # Preprocess function
    prep = keras.applications.xception.preprocess_input

    # Load dataset
    train_set = dtst.load_dataset(
        train_dir,
        target_size=args.image_size[0],
        batch_size=args.batch_size,
        preprocess_function=prep,
        augmentations=augmentations
    )
    val_set = dtst.load_dataset(
        test_dir,
        target_size=args.image_size[0],
        batch_size=args.batch_size,
        preprocess_function=prep
    )

    # Build model
    model = build_model_tricks(
        dropout=config['model']['dropout'],
        batch_norm=config['model']['batch_norm'],
        regularizer=config['model']['regularizer'],
        freeze_from=config['model']['freeze_from'],
        freeze_percent=config['model']['freeze_percent'],
    )
    return model, train_set, val_set, config


def build_and_train_optuna_model(args, report_file: str = "report_best_model.txt"):
    model, train_set, val_set, config = build_optuna_model(args, report_file)
    fit_model(model, train_set, val_set, config, log2wandb=True, save_weights=True)


def prune_model(model, train_set, val_set, config):
    print("Number of parameters in the original model: ", model.count_params())
    end_step = np.ceil(400/config["batch_size"]).astype(np.int32) * config["epochs"]
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                final_sparsity=0.50,
                                                                begin_step=0,
                                                                end_step=end_step)
    }

    # Helper function uses `prune_low_magnitude` to make only the 
    # Convolutional layers train with pruning.
    def apply_pruning_to_dense(layer):
        if "conv" in layer.name:
            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
        return layer

    # Use `tf.keras.models.clone_model` to apply `apply_pruning_to_dense` 
    # to the layers of the model.
    model_for_pruning = tf.keras.models.clone_model(
        model,
        clone_function=apply_pruning_to_dense,
    )

    # Compile model
    optimizer = utils.get_optimizer(config['optimizer']['type'],
                                config['optimizer']['learning_rate'],
                                config['optimizer']['momentum'])
    model_for_pruning.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    # Prune model
    model_for_pruning.fit(
        train_set,
        epochs=config['epochs'],
        validation_data=val_set,
        callbacks=[
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir=config['experiment_name'])
        ]
    )

    # Strip model
    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    print("Number of parameters in pruned model: ", model_for_export.count_params())

    # Evaluate model
    model_for_export.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    eval_model(model_for_export, val_set)

    return model_for_export


def prune_and_train_optuna_model(args, report_file: str = "report_best_model.txt"):
    model, train_set, val_set, config = build_optuna_model(args, report_file)

    # Load weights
    model.load_weights(args.model_weights_file)

    # Prune model
    model = prune_model(model, train_set, val_set, config)

    # Save model
    os.makedirs("out/models", exist_ok=True)
    model.save(f"out/models/{config['experiment_name']}_pruned.h5")


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


def distillation(
    args,
    train_set,
    test_set,
):

    optimizer = utils.get_optimizer(
        args.optimizer,
        args.learning_rate,
        args.momentum,
    )

    channels = [16, 32, 64, 64, 128]
    kernel_sizes = [3, 3, 3, 3]
    student = get_baseline_cnn(channels, kernel_sizes, args.image_size[0])
    # student = get_squeezenet_cnn(
    #     image_size=args.image_size[0],
    #     activation='relu',
    #     initialization='glorot_uniform',
    #     dropout=True,
    #     batch_norm=True,
    # )

    model_weights_file = './study_best_model.pkl'
    trained_teacher = build_xception_model(model_weights_file)

    temperature = 10
    alpha = 0.1

    student, teacher = distilator.train_student(
        student,
        trained_teacher,
        temperature,
        optimizer,
        train_set, test_set,
        epochs = args.epochs,
        metrics = [keras.metrics.SparseCategoricalAccuracy()],
        student_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distill_loss = keras.losses.KLDivergence(),
        alpha = alpha,
    )

    # Evaluate the student
    # ROC, AUC, Confusion Matrix, Activation Maps...



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
