from typing import Callable, Optional
from keras.preprocessing.image import ImageDataGenerator


def load_dataset(data_dir, preprocess_function: Optional[Callable] = None, augmentations: dict = {}):
    rotation_range = augmentations.get('rotation_range', 0) 
    width_shift_range = augmentations.get('width_shift_range', 0)
    height_shift_range = augmentations.get('height_shift_range', 0)
    shear_range = augmentations.get('shear_range', 0)
    zoom_range = augmentations.get('zoom_range', 0)

    data_augmentation = ImageDataGenerator(
        preprocessing_function=preprocess_function,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    target_size = (299, 299)

    data_generator = data_augmentation.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=32,
        class_mode='categorical',
        classes=['coast', 'forest', 'highway', 'inside_city',
                 'mountain', 'Opencountry', 'street', 'tallbuilding']
    )

    return data_generator


def get_n_samples(dataloader, n):
    pass
