from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np


def load_raw_volume(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads volume of skull's scan
    :param path: path to scan of skull
    :return raw_data: raw volume data of skull scan
    "return data.affine: affine transformation from skull scan
    """
    data: nib.Nifti1Image = nib.load(str(path))
    data = nib.as_closest_canonical(data)
    raw_data = data.get_fdata(caching='unchanged', dtype=np.float32)
    return raw_data, data.affine


def load_labels_volume(path: Path) -> np.ndarray:
    """
    Loads label volume from given path
    :param path: path to labeled scan of skull
    :return: raw data of label's volume
    """
    label_raw_data = load_raw_volume(path)[0].astype(np.uint8)
    return label_raw_data


def save_labels(data: np.ndarray, affine: np.ndarray, path: Path):
    """
    Saves labels in nibabel format
    :param data: 3D array with label
    :param affine: affine transformation from input nibabel image
    :param path: path to save label
    :return:
    """
    nib.save(nib.Nifti1Image(data, affine), str(path))