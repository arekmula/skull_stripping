from pathlib import Path
import tensorflow as tf


def scans_generator(scans_directory_path: Path, batch_size=16, target_size=(200, 200), shuffle=True):
    """

    :param shuffle: Whether to shuffle the data (default: True) If set to False, sorts the data in alphanumeric order.
    :param target_size: The dimensions to which all images found will be resized.
    :param scans_directory_path: It should contain one subdirectory per class. Any PNG, JPG, BMP, PPM or TIF images
     inside each of the subdirectories directory tree will be included in the generator
    :param batch_size: Size of the batches of data
    :return joined generator: A DirectoryIterator yielding tuples of (x, y) where x is a numpy array containing a batch
     of images with shape (batch_size, *target_size, channels) and y is a numpy array of corresponding labels
    """

    images_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1/255
    )

    masks_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    input_images_generator = images_datagen.flow_from_directory(directory=scans_directory_path/"scans",
                                                                target_size=target_size,
                                                                batch_size=batch_size,
                                                                seed=42,
                                                                shuffle=shuffle,
                                                                class_mode=None,
                                                                interpolation="nearest",
                                                                color_mode="grayscale")

    input_masks_generator = masks_datagen.flow_from_directory(directory=scans_directory_path/"labels",
                                                              target_size=target_size,
                                                              batch_size=batch_size,
                                                              seed=42,
                                                              shuffle=True,
                                                              class_mode=None,
                                                              interpolation="nearest")

    joined_generator = zip(input_images_generator, input_masks_generator)

    return joined_generator
