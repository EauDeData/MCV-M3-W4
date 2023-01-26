import keras
import os

from keras.applications import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model


def build_xception_model(freeze_layers: bool = True, cut_layer = None):
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
    
    os.makedirs("out", exist_ok=True)
    keras.utils.plot_model(model, to_file="out/model.png", show_shapes=False,)

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



