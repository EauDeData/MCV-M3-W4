import argparse
import keras

import dataset as dtst
import tasks
from model import build_xception_model

def __parse_args() -> argparse.Namespace:
    """
    Parses arguments.

    Returns:
        argparse.Namespace: Arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='default_train',
                        help='Task to perform: create_patches, train...')
    # Model args
    parser.add_argument('--model_config_file', type=str, default='model_configs/arquitecture1/encoder_3_layers_64_units.json',
                        help='Path to the model configuration file.')
    parser.add_argument('--model_weights_file', type=str, default=None,  # 'week3/model_weights/default.h5',
                        help='Path to the model weights file.')
    parser.add_argument('--intermediate_layer', type=str, default=None,
                        help='Name of the intermediate layer to extract features from.')
    parser.add_argument('--bottleneck_index', type = int, default = 4)
    # Dataset args
    parser.add_argument('--dataset_dir', type=str, default='data/MIT_split',
                        help='Path to the dataset directory.')
    parser.add_argument('--patches_dataset_dir', type=str, default=None,
                        help='Path to the patches dataset directory.')
    parser.add_argument('--image_size', type=int, default=[128], nargs='*',
                        help='Patch size.')
    parser.add_argument('--color_mode', type=str, default=['rgb'], nargs='*', choices=['rgb', 'grayscale'],
                        help='Color mode.')
    # Training args
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs.')
    parser.add_argument('--steps_per_epoch', type=int, default=-1,
                        help='Number of steps per epoch. If -1, it will be set to the number of samples in the dataset divided by the batch size.')
    parser.add_argument('--validation_steps', type=int, default=-1,
                        help='Number of validation steps. If -1, it will be set to the number of samples in the validation dataset divided by the batch size.')
    # Optimizer args
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum.')

    args = parser.parse_args()

    return args


def main(args: argparse.Namespace):
    pass


if __name__ == '__main__':
    args = __parse_args()
    if args.task == 'default_train':


        ### DATA ###
        train_dir = args.dataset_dir + '/train'
        test_dir = args.dataset_dir + '/test'
        prep = keras.applications.xception.preprocess_input

        train_datagen = dtst.load_dataset(train_dir, preprocess_function = prep)  
        validation_datagen = dtst.load_dataset(test_dir, preprocess_function = prep) 

        ### MODEL ###
        model = build_xception_model()

        ### TRAIN LOOP ###
        tasks.fit_model(model, train_datagen, validation_datagen, args.optimizer, args.learning_rate, args.epochs, args.momentum)
        
