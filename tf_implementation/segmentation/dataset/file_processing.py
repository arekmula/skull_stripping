import nibabel as nib
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple


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


def split_first_dataset(train_set_path: Path):
    """
    Function that splits train set from 1st dataset into train and validation set

    :param train_set_path: path to folder containing train scans where all labels and scans are in the same directory
    :return train_scans: list of tuples containing path for train scan and path for corresponding mask
    :return val_scans: list of tuples containing path for validation scan and path for corresponding mask
    """

    scan_list = []  # List of tuples containing scans and their masks
    file_extension = ".nii.gz"
    mask_partial_filename = "_mask"

    for scan_full_path in train_set_path.iterdir():
        scan_filename = scan_full_path.name[:-len(file_extension)]  # Deleting .nii.gz from file name
        if mask_partial_filename not in scan_filename:
            scan_mask_filename = scan_filename + mask_partial_filename + file_extension
            scan_mask_full_path = scan_full_path.parent / Path(scan_mask_filename)
            scan_list.append((scan_full_path, scan_mask_full_path))

    train_scans, val_scans = train_test_split(scan_list, random_state=42, train_size=0.8)

    print(f"Number of train scans from first dataset: {len(train_scans)}")
    print(f"Number of validation scans from first dataset: {len(val_scans)}")

    return train_scans, val_scans


def split_second_dataset(train_set_path: Path):
    """
    Function that splits train set from 2nd dataset into train and validation set

    :param train_set_path: path to folder where each train scan has separate folder containing scan and label
    :return train_scans: list of tuples containing path for train scan and path for corresponding mask
    :return val_scans: list of tuples containing path for validation scan and path for corresponding mask
    """

    scan_filename = "T1w.nii.gz"
    mask_filename = "mask.nii.gz"
    scan_list = []  # List of tuples containing scans and their masks

    for scan_folder_path in train_set_path.iterdir():
        scan_full_path = scan_folder_path / scan_filename
        mask_full_path = scan_folder_path / mask_filename
        scan_list.append((scan_full_path, mask_full_path))

    train_scans, val_scans = train_test_split(scan_list, random_state=42, train_size=0.8)

    print(f"Number of train scans from second dataset: {len(train_scans)}")
    print(f"Number of validation scans from second dataset: {len(val_scans)}")

    return train_scans, val_scans
