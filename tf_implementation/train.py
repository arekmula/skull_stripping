import tensorflow as tf
from tensorflow import keras

from argparse import ArgumentParser
from pathlib import Path

from segmentation.dataset import scans_generator
from segmentation.models import get_unet_model
from segmentation.losses import dice_coef_loss, dice_coef
from segmentation.utils import allow_memory_growth


def main(args):
    train_path = Path(args.train_dir_path)
    val_path = Path(args.val_dir_path)

    if args.backbone is not None:
        backbone = args.backbone
    else:
        backbone = "efficientnetb0"

    batch_size = args.batch_size

    train_generator, val_generator, train_samples, val_samples = scans_generator(train_path,
                                                                                 val_path,
                                                                                 backbone=backbone,
                                                                                 batch_size=batch_size)

    model = get_unet_model(backbone=backbone, classes=1, activation="sigmoid", encoder_weights="imagenet")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=3e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=tf.keras.metrics.BinaryAccuracy()
    )
    model.fit(
        train_generator,
        steps_per_epoch=train_samples//batch_size,
        validation_data=val_generator,
        validation_steps=val_samples//batch_size*2,
        epochs=20,
        verbose=1
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_dir_path", metavar="train_dir_path", type=str, required=True)
    parser.add_argument("--val_dir_path", metavar="val_dir_path", type=str, required=True)
    parser.add_argument("--backbone", metavar="backbone", type=str, default=None)
    parser.add_argument("--batch_size", metavar="batch_size", type=int, default=16)

    args, _ = parser.parse_known_args()

    allow_memory_growth()

    main(args)

