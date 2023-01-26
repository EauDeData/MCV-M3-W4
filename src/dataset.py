from typing import Callable, Optional
from keras.preprocessing.image import ImageDataGenerator


def load_dataset(data_dir, target_size: int = 64, batch_size: int = 16, preprocess_function: Optional[Callable] = None, augmentations: dict = {}):
    data_augmentation = ImageDataGenerator(
        preprocessing_function=preprocess_function,
        **augmentations
    )

    target_size_tuple = (target_size, target_size)

    data_generator = data_augmentation.flow_from_directory(
        data_dir,
        target_size=target_size_tuple,
        batch_size=batch_size,
        class_mode='categorical',
        classes=['coast', 'forest', 'highway', 'inside_city',
                 'mountain', 'Opencountry', 'street', 'tallbuilding']
    )

    return data_generator


def get_n_samples(dataloader, n):
    pass
