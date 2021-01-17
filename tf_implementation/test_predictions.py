import requests
import zlib

from argparse import ArgumentParser
from pathlib import Path

import nibabel as nib
import numpy as np


def main(args):
    first_dataset_predictions_path = Path(args.first_dataset_predictions_path)
    second_dataset_predictions_path = Path(args.second_dataset_predictions_path)
    url = args.url

    dice_scores = []
    for dataset_predictions_path in (first_dataset_predictions_path, second_dataset_predictions_path):
        for prediction_path in dataset_predictions_path.iterdir():
            prediction_name = prediction_path.name[:-7]  # Usuwanie '.nii.gz' z nazwy pliku
            prediction = nib.load(str(prediction_path))

            response = requests.post(f'{url}{prediction_name}', data=zlib.compress(prediction.to_bytes()))
            if response.status_code == 200:
                print(dataset_predictions_path.name, prediction_path.name, response.json())
                dice_scores.append(response.json()["dice"])
            else:
                print(f'Error processing prediction {dataset_predictions_path.name}/{prediction_name}: {response.text}')

    print(np.mean(dice_scores))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--first_dataset_predictions_path", metavar="first_dataset_predictions_path", type=str, required=True)
    parser.add_argument("--second_dataset_predictions_path", metavar="second_dataset_predictions_path", type=str, required=True)
    parser.add_argument("--url", metavar="url", type=str, required=True)

    args, _ = parser.parse_known_args()

    main(args)