import keras
import os

from keras.applications import Xception
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Convolution2D, Activation, Dropout, Concatenate
from keras.models import Model


def build_xception_model(weights = None, freeze_layers: bool = True):
    base_model = Xception(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(8, activation='softmax')(x)

    # Define the new model
    model = Model(inputs=base_model.input, outputs=x)

    # Freeze the layers of the pre-trained model
    if freeze_layers:
        for layer in base_model.layers:
            layer.trainable = False

    if weights is not None:
        model.load_weights(weights)

    os.makedirs("out", exist_ok=True)
    # keras.utils.plot_model(model, to_file="out/model.png", show_shapes=False,)

    return model


def build_xception_model_half_frozen(freeze_until: bool = 'last', freeze_from = 0):
    base_model = Xception(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(8, activation='softmax')(x)

    # Define the new model
    model = Model(inputs=base_model.input, outputs=x)
    trainable = [n for n, layer in  enumerate(base_model.layers) if sum([x in str(layer) for x in ['Conv', 'Dense', 'conv', 'dense', 'Batch']])]
    freeze = trainable[freeze_from:] if freeze_until == 'last' else trainable[freeze_from:freeze_until]

    # Freeze the layers of the pre-trained model
    for n, layer in enumerate(base_model.layers):
        if n in freeze: layer.trainable = False

    keras.utils.plot_model(model, to_file="out/model.png", show_shapes=False,)

    return model


def build_model_tricks(dropout: bool = False, regularizer: bool = False, batch_norm: bool = False, freeze_from: float = 0, freeze_percent: float = 1.0):
    base_model = Xception(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    if batch_norm: x = keras.layers.BatchNormalization()(x)
    if dropout: x = keras.layers.Dropout(.1)(x)
    x = keras.layers.ReLU()(x)
    if regularizer: x = Dense(8, activation = 'softmax', activity_regularizer=keras.regularizers.L1())(x)
    else: x = Dense(8, activation='softmax')(x)

    # Define the new model
    model = Model(inputs=base_model.input, outputs=x)
    trainable = [n for n, layer in  enumerate(base_model.layers) if sum([x in str(layer) for x in ['Conv', 'Dense', 'conv', 'dense', 'Batch']])]
    freeze_from = int(len(trainable) * freeze_from)
    freeze_until = freeze_from + int(len(trainable) * freeze_percent)
    freeze = trainable[freeze_from:freeze_until]

    # Freeze the layers of the pre-trained model
    for n, layer in enumerate(base_model.layers):
        if n in freeze: layer.trainable = False

    # keras.utils.plot_model(model, to_file="out/model.png", show_shapes=False,)

    return model


def get_intermediate_layer_model(model, layer_index: int = -1) -> Model:
    """
    Returns the intermediate layers model from a given model. 

    Args:
        model (Sequential): Keras model.
        layer_name (int): Index of the layer to be returned.
    """
    intermediate_layer_model = Model(inputs=model.input,
                                        outputs=model.layers[layer_index].output)
    return intermediate_layer_model


def get_baseline_cnn(channels, kernel_sizes, image_size, weights=None):

    input_layer = keras.Input(shape=(image_size, image_size, 3))

    x = Conv2D(channels[0], (kernel_sizes[0], kernel_sizes[0]), activation='relu')(input_layer)
    for ch, ks in zip(channels[1:-1], kernel_sizes[1:-1]):
        x = Conv2D(ch, (ks, ks), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
    x = Conv2D(channels[-1], (kernel_sizes[-1], kernel_sizes[-1]), activation='relu')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(8, activation='softmax')(x)

    # x = Flatten()(x)
    # x = Dense(64, activation='relu')(x)
    # x = Dense(8, activation='softmax')(x)

    model = Model(
        inputs=input_layer,
        outputs=x,
    )

    os.makedirs("out", exist_ok=True)
    if weights is not None:
        model.load_weights(weights)

    # keras.utils.plot_model(model, to_file="out/model.png", show_shapes=False,)

    return model


def small_squeezenet_cnn(
    image_size: int, activation: str, initialization: str, dropout: float, batch_norm: bool,
):
    input_img = keras.Input(shape=(image_size, image_size, 3))
    conv1 = Convolution2D(
        96, (7, 7), activation=activation, kernel_initializer=initialization,  # glorot_uniform,
        strides=(2, 2), padding='same', name='conv1',
    )(input_img)
    maxpool1 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool1',
    )(conv1)




    fire2_squeeze = Convolution2D(
        16, (1, 1), activation=None, kernel_initializer=initialization,
        padding='same', name='fire2_squeeze',
        )(maxpool1)
    fire2_expand1 = Convolution2D(
        64, (1, 1), activation=None, kernel_initializer=initialization,
        padding='same', name='fire2_expand1',
        )(fire2_squeeze)
    if batch_norm:
        fire2_expand1 = BatchNormalization()(fire2_expand1)
    fire2_expand1 = Activation(activation)(fire2_expand1)
    if dropout:
        fire2_expand1 = Dropout(0.1,)(fire2_expand1)
    fire2_expand2 = Convolution2D(
        64, (3, 3), activation=None, kernel_initializer=initialization,
        padding='same', name='fire2_expand2',
        )(fire2_squeeze)
    if batch_norm:
        fire2_expand2 = BatchNormalization()(fire2_expand2)
    fire2_expand2 = Activation(activation)(fire2_expand2)
    if dropout:
        fire2_expand2 = Dropout(0.1,)(fire2_expand2)
    merge2 = Concatenate(axis=1)([fire2_expand1, fire2_expand2])

    fire3_squeeze = Convolution2D(
        16, (1, 1), activation=None, kernel_initializer=initialization,
        padding='same', name='fire3_squeeze',
        )(merge2)
    fire3_expand1 = Convolution2D(
        64, (1, 1), activation=None, kernel_initializer=initialization,
        padding='same', name='fire3_expand1',
        )(fire3_squeeze)
    if batch_norm:
        fire3_expand1 = BatchNormalization()(fire3_expand1)
    fire3_expand1 = Activation(activation)(fire3_expand1)
    if dropout:
        fire3_expand1 = Dropout(0.1,)(fire3_expand1)
    fire3_expand2 = Convolution2D(
        64, (3, 3), activation=None, kernel_initializer=initialization,
        padding='same', name='fire3_expand2',
        )(fire3_squeeze)
    if batch_norm:
        fire3_expand2 = BatchNormalization()(fire3_expand2)
    fire3_expand2 = Activation(activation)(fire3_expand2)
    if dropout:
        fire3_expand2 = Dropout(0.1,)(fire3_expand2)
    merge3 = Concatenate(axis=1)([fire3_expand1, fire3_expand2])

    fire4_squeeze = Convolution2D(
        32, (1, 1), activation=None, kernel_initializer=initialization,
        padding='same', name='fire4_squeeze',
        )(merge3)
    fire4_expand1 = Convolution2D(
        128, (1, 1), activation=None, kernel_initializer=initialization,
        padding='same', name='fire4_expand1',
        )(fire4_squeeze)
    if batch_norm:
        fire4_expand1 = BatchNormalization()(fire4_expand1)
    fire4_expand1 = Activation(activation)(fire4_expand1)
    if dropout:
        fire4_expand1 = Dropout(0.1,)(fire4_expand1)
    fire4_expand2 = Convolution2D(
        128, (3, 3), activation=None, kernel_initializer=initialization,
        padding='same', name='fire4_expand2',
        )(fire4_squeeze)
    if batch_norm:
        fire4_expand2 = BatchNormalization()(fire4_expand2)
    fire4_expand2 = Activation(activation)(fire4_expand2)
    if dropout:
        fire4_expand2 = Dropout(0.1,)(fire4_expand2)
    merge4 = Concatenate(axis=1)([fire4_expand1, fire4_expand2])
    maxpool4 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool4',
        )(merge4)

    fire5_squeeze = Convolution2D(
        32, (1, 1), activation=None, kernel_initializer=initialization,
        padding='same', name='fire5_squeeze',
        )(maxpool4)
    fire5_expand1 = Convolution2D(
        128, (1, 1), activation=None, kernel_initializer=initialization,
        padding='same', name='fire5_expand1',
        )(fire5_squeeze)
    if batch_norm:
        fire5_expand1 = BatchNormalization()(fire5_expand1)
    fire5_expand1 = Activation(activation)(fire5_expand1)
    if dropout:
        fire5_expand1 = Dropout(0.1,)(fire5_expand1)
    fire5_expand2 = Convolution2D(
        128, (3, 3), activation=None, kernel_initializer=initialization,
        padding='same', name='fire5_expand2',
        )(fire5_squeeze)
    if batch_norm:
        fire5_expand2 = BatchNormalization()(fire5_expand2)
    fire5_expand2 = Activation(activation)(fire5_expand2)
    if dropout:
        fire5_expand2 = Dropout(0.1,)(fire5_expand2)
    merge5 = Concatenate(axis=1)([fire5_expand1, fire5_expand2])


    fire9_dropout = Dropout(0.5, name='fire9_dropout')(merge5)
    conv10 = Convolution2D(
        8, (1, 1), activation=None, kernel_initializer=initialization,
        padding='valid', name='conv10',
        )(fire9_dropout)

    global_avgpool10 = GlobalAveragePooling2D()(conv10)
    softmax = Dense(8, activation='softmax', name='softmax')(global_avgpool10)

    return Model(inputs=input_img, outputs=softmax)


def get_squeezenet_cnn(
    image_size: int, activation: str, initialization: str, dropout: float, batch_norm: bool
):
    input_img = keras.Input(shape=(image_size, image_size, 3))
    conv1 = Convolution2D(
        96, (7, 7), activation=activation, kernel_initializer=initialization,  # glorot_uniform,
        strides=(2, 2), padding='same', name='conv1',
    )(input_img)
    maxpool1 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool1',
    )(conv1)

    fire2_squeeze = Convolution2D(
        16, (1, 1), activation=None, kernel_initializer=initialization,
        padding='same', name='fire2_squeeze',
        )(maxpool1)
    fire2_expand1 = Convolution2D(
        64, (1, 1), activation=None, kernel_initializer=initialization,
        padding='same', name='fire2_expand1',
        )(fire2_squeeze)
    if batch_norm:
        fire2_expand1 = BatchNormalization()(fire2_expand1)
    fire2_expand1 = Activation(activation)(fire2_expand1)
    if dropout:
        fire2_expand1 = Dropout(0.1,)(fire2_expand1)
    fire2_expand2 = Convolution2D(
        64, (3, 3), activation=None, kernel_initializer=initialization,
        padding='same', name='fire2_expand2',
        )(fire2_squeeze)
    if batch_norm:
        fire2_expand2 = BatchNormalization()(fire2_expand2)
    fire2_expand2 = Activation(activation)(fire2_expand2)
    if dropout:
        fire2_expand2 = Dropout(0.1,)(fire2_expand2)
    merge2 = Concatenate(axis=1)([fire2_expand1, fire2_expand2])

    fire3_squeeze = Convolution2D(
        16, (1, 1), activation=None, kernel_initializer=initialization,
        padding='same', name='fire3_squeeze',
        )(merge2)
    fire3_expand1 = Convolution2D(
        64, (1, 1), activation=None, kernel_initializer=initialization,
        padding='same', name='fire3_expand1',
        )(fire3_squeeze)
    if batch_norm:
        fire3_expand1 = BatchNormalization()(fire3_expand1)
    fire3_expand1 = Activation(activation)(fire3_expand1)
    if dropout:
        fire3_expand1 = Dropout(0.1,)(fire3_expand1)
    fire3_expand2 = Convolution2D(
        64, (3, 3), activation=None, kernel_initializer=initialization,
        padding='same', name='fire3_expand2',
        )(fire3_squeeze)
    if batch_norm:
        fire3_expand2 = BatchNormalization()(fire3_expand2)
    fire3_expand2 = Activation(activation)(fire3_expand2)
    if dropout:
        fire3_expand2 = Dropout(0.1,)(fire3_expand2)
    merge3 = Concatenate(axis=1)([fire3_expand1, fire3_expand2])

    fire4_squeeze = Convolution2D(
        32, (1, 1), activation=None, kernel_initializer=initialization,
        padding='same', name='fire4_squeeze',
        )(merge3)
    fire4_expand1 = Convolution2D(
        128, (1, 1), activation=None, kernel_initializer=initialization,
        padding='same', name='fire4_expand1',
        )(fire4_squeeze)
    if batch_norm:
        fire4_expand1 = BatchNormalization()(fire4_expand1)
    fire4_expand1 = Activation(activation)(fire4_expand1)
    if dropout:
        fire4_expand1 = Dropout(0.1,)(fire4_expand1)
    fire4_expand2 = Convolution2D(
        128, (3, 3), activation=None, kernel_initializer=initialization,
        padding='same', name='fire4_expand2',
        )(fire4_squeeze)
    if batch_norm:
        fire4_expand2 = BatchNormalization()(fire4_expand2)
    fire4_expand2 = Activation(activation)(fire4_expand2)
    if dropout:
        fire4_expand2 = Dropout(0.1,)(fire4_expand2)
    merge4 = Concatenate(axis=1)([fire4_expand1, fire4_expand2])
    maxpool4 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool4',
        )(merge4)

    fire5_squeeze = Convolution2D(
        32, (1, 1), activation=None, kernel_initializer=initialization,
        padding='same', name='fire5_squeeze',
        )(maxpool4)
    fire5_expand1 = Convolution2D(
        128, (1, 1), activation=None, kernel_initializer=initialization,
        padding='same', name='fire5_expand1',
        )(fire5_squeeze)
    if batch_norm:
        fire5_expand1 = BatchNormalization()(fire5_expand1)
    fire5_expand1 = Activation(activation)(fire5_expand1)
    if dropout:
        fire5_expand1 = Dropout(0.1,)(fire5_expand1)
    fire5_expand2 = Convolution2D(
        128, (3, 3), activation=None, kernel_initializer=initialization,
        padding='same', name='fire5_expand2',
        )(fire5_squeeze)
    if batch_norm:
        fire5_expand2 = BatchNormalization()(fire5_expand2)
    fire5_expand2 = Activation(activation)(fire5_expand2)
    if dropout:
        fire5_expand2 = Dropout(0.1,)(fire5_expand2)
    merge5 = Concatenate(axis=1)([fire5_expand1, fire5_expand2])

    fire6_squeeze = Convolution2D(
        48, (1, 1), activation=None, kernel_initializer=initialization,
        padding='same', name='fire6_squeeze',
        )(merge5)
    fire6_expand1 = Convolution2D(
        192, (1, 1), activation=None, kernel_initializer=initialization,
        padding='same', name='fire6_expand1',
        )(fire6_squeeze)
    if batch_norm:
        fire6_expand1 = BatchNormalization()(fire6_expand1)
    fire6_expand1 = Activation(activation)(fire6_expand1)
    if dropout:
        fire6_expand1 = Dropout(0.1,)(fire6_expand1)
    fire6_expand2 = Convolution2D(
        192, (3, 3), activation=None, kernel_initializer=initialization,
        padding='same', name='fire6_expand2',
        )(fire6_squeeze)
    if batch_norm:
        fire6_expand2 = BatchNormalization()(fire6_expand2)
    fire6_expand2 = Activation(activation)(fire6_expand2)
    if dropout:
        fire6_expand2 = Dropout(0.1,)(fire6_expand2)
    merge6 = Concatenate(axis=1)([fire6_expand1, fire6_expand2])

    fire7_squeeze = Convolution2D(
        48, (1, 1), activation=None, kernel_initializer=initialization,
        padding='same', name='fire7_squeeze',
        )(merge6)
    fire7_expand1 = Convolution2D(
        192, (1, 1), activation=None, kernel_initializer=initialization,
        padding='same', name='fire7_expand1',
        )(fire7_squeeze)
    if batch_norm:
        fire7_expand1 = BatchNormalization()(fire7_expand1)
    fire7_expand1 = Activation(activation)(fire7_expand1)
    if dropout:
        fire7_expand1 = Dropout(0.1,)(fire7_expand1)
    fire7_expand2 = Convolution2D(
        192, (3, 3), activation=None, kernel_initializer=initialization,
        padding='same', name='fire7_expand2',
        )(fire7_squeeze)
    if batch_norm:
        fire7_expand2 = BatchNormalization()(fire7_expand2)
    fire7_expand2 = Activation(activation)(fire7_expand2)
    if dropout:
        fire7_expand2 = Dropout(0.1,)(fire7_expand2)
    merge7 = Concatenate(axis=1)([fire7_expand1, fire7_expand2])

    fire8_squeeze = Convolution2D(
        64, (1, 1), activation=None, kernel_initializer=initialization,
        padding='same', name='fire8_squeeze',
        )(merge7)
    fire8_expand1 = Convolution2D(
        256, (1, 1), activation=None, kernel_initializer=initialization,
        padding='same', name='fire8_expand1',
        )(fire8_squeeze)
    if batch_norm:
        fire8_expand1 = BatchNormalization()(fire8_expand1)
    fire8_expand1 = Activation(activation)(fire8_expand1)
    if dropout:
        fire8_expand1 = Dropout(0.1,)(fire8_expand1)
    fire8_expand2 = Convolution2D(
        256, (3, 3), activation=None, kernel_initializer=initialization,
        padding='same', name='fire8_expand2',
        )(fire8_squeeze)
    if batch_norm:
        fire8_expand2 = BatchNormalization()(fire8_expand2)
    fire8_expand2 = Activation(activation)(fire8_expand2)
    if dropout:
        fire8_expand2 = Dropout(0.1,)(fire8_expand2)
    merge8 = Concatenate(axis=1)([fire8_expand1, fire8_expand2])

    maxpool8 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool8',
        )(merge8)
    fire9_squeeze = Convolution2D(
        64, (1, 1), activation=None, kernel_initializer=initialization,
        padding='same', name='fire9_squeeze',
        )(maxpool8)
    fire9_expand1 = Convolution2D(
        256, (1, 1), activation=None, kernel_initializer=initialization,
        padding='same', name='fire9_expand1',
        )(fire9_squeeze)
    if batch_norm:
        fire9_expand1 = BatchNormalization()(fire9_expand1)
    fire9_expand1 = Activation(activation)(fire9_expand1)
    if dropout:
        fire9_expand1 = Dropout(0.1,)(fire9_expand1)
    fire9_expand2 = Convolution2D(
        256, (3, 3), activation=None, kernel_initializer=initialization,
        padding='same', name='fire9_expand2',
        )(fire9_squeeze)
    if batch_norm:
        fire9_expand2 = BatchNormalization()(fire9_expand2)
    fire9_expand2 = Activation(activation)(fire9_expand2)
    if dropout:
        fire9_expand2 = Dropout(0.1,)(fire9_expand2)
    merge9 = Concatenate(axis=1)([fire9_expand1, fire9_expand2])

    fire9_dropout = Dropout(0.5, name='fire9_dropout')(merge9)
    conv10 = Convolution2D(
        8, (1, 1), activation=None, kernel_initializer=initialization,
        padding='valid', name='conv10',
        )(fire9_dropout)

    global_avgpool10 = GlobalAveragePooling2D()(conv10)
    softmax = Activation("softmax", name='softmax')(global_avgpool10)

    return Model(inputs=input_img, outputs=softmax)
