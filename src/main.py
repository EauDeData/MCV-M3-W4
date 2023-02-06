import os
import argparse
import keras
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import dataset as dtst
import tasks
from model import build_xception_model, build_xception_model_half_frozen, build_model_tricks


def __parse_args() -> argparse.Namespace:
    """
    Parses arguments.

    Returns:
        argparse.Namespace: Arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='optuna_train',
                        help='Task to perform: [default_train, evaluate, cross_validation, optuna_search, optuna_train]')
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
    parser.add_argument('--train_augmentations_file', type=str, default=None, #"configs/augmentations/train_augmentations.json",
                        help='Path to the train augmentations file.')
    # Training args
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size.')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs.')
    # parser.add_argument('--steps_per_epoch', type=int, default=-1,
    #                     help='Number of steps per epoch. If -1, it will be set to the number of samples in the dataset divided by the batch size.')
    # parser.add_argument('--validation_steps', type=int, default=-1,
    #                     help='Number of validation steps. If -1, it will be set to the number of samples in the validation dataset divided by the batch size.')
    parser.add_argument('--log2wandb', type=bool, default=True,
                        help='Log to wandb.')
    # Optimizer args
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer.')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum.')
    # Optuna args
    parser.add_argument('--n_trials', type=int, default=4,
                        help='Number of trials for the optuna search.')

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
        tasks.optuna_search(args, args.dataset_dir, args.n_trials)

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
        model = build_model_tricks(dropout=False, batch_norm=False, regularizer=True)

        ### TRAIN LOOP ###
        tasks.train_properly_implemented(model, train_datagen, validation_datagen, args.optimizer, args.learning_rate, args.epochs, args.momentum)
    
    elif args.task == 'train_tricks_train':
        
        ### DATA ###
        train_dir = args.dataset_dir + '/train'
        test_dir = args.dataset_dir + '/test'
        prep = keras.applications.xception.preprocess_input

        train_datagen = dtst.load_dataset(train_dir, preprocess_function = prep)  
        validation_datagen = dtst.load_dataset(test_dir, preprocess_function = prep) 

        ### MODEL ###
        model = build_model_tricks(dropout=False, batch_norm=False, regularizer=True)

        ### TRAIN LOOP ###
        tasks.train_tricks_train(model, train_datagen, validation_datagen, args.optimizer, args.learning_rate, args.epochs, 0.1, args.momentum)

    elif args.task == 'optuna_train':
        tasks.build_and_train_optuna_model(args)

    elif args.task == 'results_vis':
        
        sns.set()
        args.epochs = 1
        model, train_set, val_set = tasks.build_optuna_model(args)
        model.load_weights('out/model_weights/xception_best_model_20230129-145950.h5')
        classes = (val_set.class_indices)

        #model = build_xception_model(weights = './out/model_weights/xception_20230128-221841.h5')

        # make predictions on the test data
        from keras.preprocessing.image import ImageDataGenerator

        test_steps = val_set.n // args.batch_size
        y_score = []
        test_labels = []
        predicted_labels = []
        labels_one_hot = []
        string_labels = sorted(list(classes.keys()), key = lambda x: classes[x])
        y_score_single = []
        y_labels_argmax = []

        for n, (images, label_batch) in enumerate(val_set):

            for image, label in zip(images, label_batch):
                prediction = model(image[None, :, :, :])[0]
                labels_one_hot.append(label)
                y_labels_argmax.append(np.argmax(label))
                test_labels.append(string_labels[np.argmax(label)])
                y_score.append(prediction)
                predicted_labels.append(string_labels[np.argmax(prediction)])
                y_score_single.append(prediction[np.argmax(label)])


            if n == test_steps:
                break
        #print(len(y_score), len(predicted_labels), len(test_labels))
        y_score = np.array(y_score)
        import scikitplot as skplt
        skplt.metrics.plot_roc(y_labels_argmax, y_score)

        plt.savefig('ROCK!!!.png')
        plt.clf()

        skplt.metrics.plot_precision_recall(y_labels_argmax, y_score)
        plt.savefig('UNA PR!!!.png')
        plt.clf()

        


        # Prepare data for confusion matrix
        axs_dict = classes
        cat = len(axs_dict)
        matrix = np.zeros((cat, cat))
        for gt, pred in zip(test_labels, predicted_labels):
            matrix[axs_dict[gt], axs_dict[pred]] += 1

        matrixrel = np.zeros((cat, cat))
        for x in test_labels:
            for y in predicted_labels:
                matrixrel[axs_dict[x], axs_dict[y]] = round(100 * matrix[axs_dict[x], axs_dict[y]] / matrix[axs_dict[x],].sum())
                
        # Plot confusion matrix

        cmap = sns.cubehelix_palette(start=1.6, light=0.8, as_cmap=True,)
        fig, axs = plt.subplots(ncols = 2, figsize = (10, 4))
        ax = axs[0]

        sns.heatmap(matrix.astype(int), annot=True, cmap = cmap, ax = ax, cbar = False)
        ax.set_ylabel('GT')
        ax.set_xlabel("Predicted")


        ax.set_title("Absolute count")
        ax.set_xticks(list(range(len(axs_dict))), rotation = 45) # Rotates X-Axis Ticks by 45-degrees
        ax.set_yticks(list(range(len(axs_dict))), rotation = 45) # Rotates X-Axis Ticks by 45-degrees


        ax.set_yticklabels(axs_dict, rotation = 20)
        ax.set_xticklabels(axs_dict, rotation = 45)


        sns.heatmap(matrixrel, annot=True, cmap = cmap, ax = axs[1], cbar = False, )
        ax = axs[1]
        ax.set_title("Relative count (%)")
        ax.set_yticklabels([], rotation = 0)
        ax.set_xticklabels(axs_dict, rotation = 45)
        ax.set_xlabel("Predicted")

        fig.suptitle('Confusion matrix for test set predictions')
        plt.savefig('MATRIX.png')

    elif args.task == "distil":
        # TODO: implement distillation
        channels = [16, 32, 64, 64]
        kernel_sizes = [3, 3, 3, 3]
        student = get_baseline_cnn(channels, kernel_sizes, args.image_size[0])
        # teacher = model_totxo()
        ### DATA ###
        train_dir = args.dataset_dir + '/train'
        test_dir = args.dataset_dir + '/test'
        prep = keras.applications.xception.preprocess_input  # TODO: triar preprocessat. Si agafem el d'una altra arquitectura preentrenada, caldrà fer un preprocessat diferent

        train_datagen = dtst.load_dataset(train_dir, preprocess_function = prep)
        validation_datagen = dtst.load_dataset(test_dir, preprocess_function = prep)

        ### TRAIN LOOP ###
        tasks.train_properly_implemented(student, train_datagen, validation_datagen, args.optimizer, args.learning_rate, args.epochs, args.momentum)
        # tasks.distillation(teacher, student, ...)  # TODO: implement distillation


if __name__ == '__main__':
    args = __parse_args()
    main(args)
