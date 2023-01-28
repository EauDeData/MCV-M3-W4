import os
import argparse
import keras
import datetime
import numpy as np
from typing import Dict, Any, List

import dataset as dtst
import tasks
import utils
from model import build_xception_model, build_xception_model_half_frozen, build_model_tricks


def __parse_args() -> argparse.Namespace:
    """
    Parses arguments.

    Returns:
        argparse.Namespace: Arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='optuna_search',
                        help='Task to perform: [default_train, evaluate, cross_validation, optuna_search]')
    # Model args
    # parser.add_argument('--model_config_file', type=str, default='model_configs/arquitecture1/encoder_3_layers_64_units.json',
    #                     help='Path to the model configuration file.')
    parser.add_argument('--model_weights_file', type=str, default=None,  # 'week3/model_weights/default.h5',
                        help='Path to the model weights file.')
    # parser.add_argument('--intermediate_layer', type=str, default=None,
    #                     help='Name of the intermediate layer to extract features from.')
    # parser.add_argument('--bottleneck_index', type=int, default=4)
    # Dataset args
    parser.add_argument('--dataset_dir', type=str, default='data/MIT_split',
                        help='Path to the dataset directory.')
    parser.add_argument('--patches_dataset_dir', type=str, default=None,
                        help='Path to the patches dataset directory.')
    parser.add_argument('--image_size', type=int, default=[224], nargs='*',
                        help='Patch size.')
    # parser.add_argument('--color_mode', type=str, default=['rgb'], nargs='*', choices=['rgb', 'grayscale'],
    #                     help='Color mode.')
    parser.add_argument('--train_augmentations_file', type=str, default="configs/augmentations/train_augmentations.json",
                        help='Path to the train augmentations file.')
    # Training args
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs.')
    # parser.add_argument('--steps_per_epoch', type=int, default=-1,
    #                     help='Number of steps per epoch. If -1, it will be set to the number of samples in the dataset divided by the batch size.')
    # parser.add_argument('--validation_steps', type=int, default=-1,
    #                     help='Number of validation steps. If -1, it will be set to the number of samples in the validation dataset divided by the batch size.')
    parser.add_argument('--log2wandb', type=bool, default=False,
                        help='Log to wandb.')
    # Optimizer args
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer.')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum.')

    args = parser.parse_args()

    return args


def main(args: argparse.Namespace):
    if args.task == 'default_train':
        tasks.default_train(args, args.dataset_dir, log2wandb=args.log2wandb)

    elif args.task == 'evaluate':
        model = build_xception_model(args.model_weights_file)

        prep = keras.applications.xception.preprocess_input
        validation_datagen = dtst.load_dataset(
        args.dataset_dir + "/test",
        target_size=args.image_size[0],
        batch_size=args.batch_size,
        preprocess_function=prep
        )

        tasks.eval_model(model, validation_datagen)

    elif args.task == 'optuna_search':
        tasks.optuna_search(args, args.dataset_dir)

    elif args.task == 'cross_validation':
        # List the directories in the dataset directory
        dataset_dir = args.dataset_dir
        dataset_dirs = [os.path.join(dataset_dir, d)
                        for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        dataset_dirs.sort()

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # Train the model for each fold and save the metrics, then compute the mean and std
        loss = []
        accuracy = []
        
        for i, dataset_dir in enumerate(dataset_dirs):
            print(f'Cross validation {i + 1}/{len(dataset_dirs)}')
            experiment_name = f'xception_fold{i+1}_{timestamp}'
            metrics = tasks.default_train(args, dataset_dir, experiment_name, log2wandb=args.log2wandb, save_weights=False)
            loss.append(metrics[0])
            accuracy.append(metrics[1])

        loss = np.array(loss)
        accuracy = np.array(accuracy)

        print(f'Loss: {loss.mean()} +- {loss.std()}')
        print(f'Accuracy: {accuracy.mean()} +- {accuracy.std()}')
    
    elif args.task == 'train_frozen':


        ### DATA ###
        train_dir = args.dataset_dir + '/train'
        test_dir = args.dataset_dir + '/test'
        prep = keras.applications.xception.preprocess_input

        train_datagen = dtst.load_dataset(train_dir, preprocess_function = prep)  
        validation_datagen = dtst.load_dataset(test_dir, preprocess_function = prep) 

        ### MODEL ###
        model = build_xception_model_half_frozen(freeze_from=20, freeze_until=60)

        ### TRAIN LOOP ###
        tasks.train_properly_implemented(model, train_datagen, validation_datagen, args.optimizer, args.learning_rate, args.epochs, args.momentum)

    elif args.task == 'train_tricks_arch':
        
        ### DATA ###
        train_dir = args.dataset_dir + '/train'
        test_dir = args.dataset_dir + '/test'
        prep = keras.applications.xception.preprocess_input

        train_datagen = dtst.load_dataset(train_dir, preprocess_function = prep)  
        validation_datagen = dtst.load_dataset(test_dir, preprocess_function = prep) 

        ### MODEL ###
        model = build_model_tricks(dropout=True)

        ### TRAIN LOOP ###
        tasks.train_properly_implemented(model, train_datagen, validation_datagen, args.optimizer, args.learning_rate, args.epochs, args.momentum)


if __name__ == '__main__':
    args = __parse_args()
    main(args)
    

        