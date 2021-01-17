import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

from argparse import ArgumentParser
from pathlib import Path


from segmentation.dataset import test_scans_generator, load_affine, save_labels
from segmentation.losses import dice_loss
from segmentation.metrics import f_score
from segmentation.utils import allow_memory_growth


def main(args):
    predictions_base_path = Path(args.predictions_path)
    first_dataset_predictions_path = predictions_base_path / "first"
    second_dataset_predictions_path = predictions_base_path / "second"

    first_dataset_predictions_path.mkdir(exist_ok=True, parents=True)
    second_dataset_predictions_path.mkdir(exist_ok=True, parents=True)

    first_dataset_test_path = Path(args.first_dataset_test_path)
    second_dataset_test_path = Path(args.second_dataset_test_path)

    dice_score = f_score(threshold=0.5)

    model = keras.models.load_model(args.model_path, custom_objects={"f1-score": dice_score, "dice_loss": dice_loss})
    # print(model.summary())

    for scan_path in first_dataset_test_path.iterdir():
        print(f"Processing scan: {scan_path.name}")
        test_scans, test_samples = test_scans_generator(scan_path)
        affine = load_affine(affine_path=scan_path / str(scan_path.name + ".npy"))

        with open(str(scan_path/scan_path.name) + ".json") as json_file:
            shape_dict = json.load(json_file)
        scan_shape = (shape_dict["x"], shape_dict["y"], shape_dict["z"])
        labels = np.zeros(scan_shape, dtype=np.uint8)

        for sample in range(test_samples):
            image = next(test_scans)
            prediction = model.predict(image)[0]
            prediction = np.where(prediction > 0.5, np.uint8(1), np.uint8(0))
            prediction_resized = cv2.resize(prediction, (scan_shape[2], scan_shape[1]))
            labels[sample, :, :] = prediction_resized.squeeze()

        save_labels(labels, affine, first_dataset_predictions_path / f"{scan_path.name}.nii.gz")

    for scan_path in second_dataset_test_path.iterdir():
        print(f"Processing scan {scan_path.name}")
        test_scans, test_samples = test_scans_generator(scan_path)
        affine = load_affine(affine_path=scan_path / str(scan_path.name + ".npy"))

        with open(str(scan_path/scan_path.name) + ".json") as json_file:
            shape_dict = json.load(json_file)
        scan_shape = (shape_dict["x"], shape_dict["y"], shape_dict["z"])
        labels = np.zeros(scan_shape, dtype=np.uint8)

        for sample in range(test_samples):
            image = next(test_scans)
            prediction = model.predict(image)[0]
            prediction = np.where(prediction > 0.5, np.uint8(1), np.uint8(0))
            prediction_resized = cv2.resize(prediction, (scan_shape[2], scan_shape[1]))
            labels[sample, :, :] = prediction_resized.squeeze()

        save_labels(labels, affine, second_dataset_predictions_path / f"{scan_path.name}.nii.gz")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--predictions_path", metavar="predictions_path", type=str, required=True)
    parser.add_argument("--first_dataset_test_path", metavar="first_dataset_test_path", type=str, required=True)
    parser.add_argument("--second_dataset_test_path", metavar="second_dataset_test_path", type=str, required=True)
    parser.add_argument("--model_path", metavar="model_path", type=str, required=True)

    args, _ = parser.parse_known_args()

    allow_memory_growth()

    main(args)

