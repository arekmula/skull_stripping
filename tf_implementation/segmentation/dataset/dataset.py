import tensorflow as tf

from pathlib import Path
from segmentation_models import get_preprocessing


def scans_generator(scans_train_directory_path: Path, scans_val_directory_path: Path, batch_size=16,
                    target_size=(200, 200), shuffle=True, backbone="efficientnetb0"):
    """
    Tensorflow dataset generator.

    :param scans_train_directory_path: It should contain one subdirectory per class. Any PNG, JPG, BMP, PPM or TIF
    images inside each of the subdirectories directory tree will be included in the generator.
    :param scans_val_directory_path: It should contain one subdirectory per class. Any PNG, JPG, BMP, PPM or TIF
    images inside each of the subdirectories directory tree will be included in the generator.
    :param backbone: backbone name of network you will use
    :param shuffle: Whether to shuffle the data (default: True) If set to False, sorts the data in alphanumeric order.
    :param target_size: The dimensions to which all images found will be resized.
    :param batch_size: Size of the batches of data
    :return train_combined_generator: A DirectoryIterator yielding tuples of (x, y) where x is a numpy array containing
    a batch of images with shape (batch_size, *target_size, channels) and y is a numpy array of corresponding labels
    :return val_combined_generator: A DirectoryIterator yielding tuples of (x, y) where x is a numpy array containing
    a batch of images with shape (batch_size, *target_size, channels) and y is a numpy array of corresponding labels
    """

    # Preprocess image based on backbone implementation
    preprocess_input = get_preprocessing(backbone)

    # TODO: Check augmentations
    images_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        # rescale=1/255, # -> Uncomment if not using preprocess_input
        preprocessing_function=preprocess_input
    )

    masks_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    train_images_generator = images_datagen.flow_from_directory(directory=scans_train_directory_path / "scans",
                                                                target_size=target_size,
                                                                batch_size=batch_size,
                                                                seed=42,
                                                                shuffle=shuffle,
                                                                class_mode=None,
                                                                interpolation="bilinear",
                                                                )

    train_masks_generator = masks_datagen.flow_from_directory(directory=scans_train_directory_path / "labels",
                                                              target_size=target_size,
                                                              batch_size=batch_size,
                                                              seed=42,
                                                              shuffle=True,
                                                              class_mode=None,
                                                              interpolation="nearest",
                                                              )

    val_images_generator = images_datagen.flow_from_directory(directory=scans_val_directory_path / "scans",
                                                              target_size=target_size,
                                                              batch_size=batch_size,
                                                              seed=42,
                                                              shuffle=shuffle,
                                                              class_mode=None,
                                                              interpolation="bilinear",
                                                              )

    val_masks_generator = masks_datagen.flow_from_directory(directory=scans_val_directory_path / "labels",
                                                            target_size=target_size,
                                                            batch_size=batch_size,
                                                            seed=42,
                                                            shuffle=True,
                                                            class_mode=None,
                                                            interpolation="nearest",
                                                            )

    train_combined_generator = zip(train_images_generator, train_masks_generator)
    val_combined_generator = zip(val_images_generator, val_masks_generator)

    return train_combined_generator, val_combined_generator
