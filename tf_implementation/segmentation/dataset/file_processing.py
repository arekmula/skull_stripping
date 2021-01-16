import nibabel as nib
import numpy as np
from PIL import Image

from matplotlib import pyplot as plt
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
    Splits train set from 1st dataset into train and validation set

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
    Splits train set from 2nd dataset into train and validation set

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


def save_scan_to_xyz_slices(scan: Tuple, save_path: Path, scan_number: int, axis="x"):
    """
    Splits scan and scan's label to separate x, y, z slices and then saves it to separate files.

    :param axis: axis to split
    :param scan_number: scan number to store
    :param scan: Tuple containing path to scan and path to corresponding label
    :param save_path: Path to folder where slices will be stored. The folder should have following structure:
    main folder/
                affine/
                x/
                    labels/images/
                    scans/images/
                y/
                    labels/images/
                    scans/images/
                z/
                    labels/images/
                    scans/images/

    :return:
    """
    file_extension = ".nii.gz"
    scan_name = scan[0].name[:-len(file_extension)] + "_" + str(scan_number)

    raw_volume, affine = load_raw_volume(scan[0])
    mask_volume = load_labels_volume(scan[1])

    x_scans = get_axes_slices_from_volume(raw_volume=raw_volume, axis=axis)
    x_scans = normalize_slice_values(x_scans)
    x_labels = get_axes_slices_from_volume(raw_volume=mask_volume, axis=axis)

    print(f"\r Saving scan: {scan_name}")
    for scan_number, (scan, label) in enumerate(zip(x_scans, x_labels)):
        path = save_path / Path(axis) / Path("scans/images") / Path(scan_name + str(scan_number) + ".png")
        save_image_to_png(scan, path)

        path = save_path / Path(axis) / Path("labels/images") / Path(scan_name + str(scan_number) + ".png")
        save_image_to_png(label, path)

    path = save_path / Path("affine") / Path(scan_name)
    np.save(path, affine)


def get_axes_slices_from_volume(raw_volume: np.ndarray, axis="x"):
    """
    Returns slices from given axis

    :param axis: axis from which return slice
    :param raw_volume: Scan volume
    :return: list of slices from given axis
    """

    if axis == "x":
        x_slices = []
        for current_slice in range(raw_volume.shape[0]):
            x_slices.append(raw_volume[current_slice])

        return x_slices

    elif axis == "y":
        y_slices = []
        for current_slice in range(raw_volume.shape[1]):
            y_slices.append(raw_volume[:, current_slice])

        return y_slices

    elif axis == "z":
        z_slices = []
        for current_slice in range(raw_volume.shape[2]):
            z_slices.append(raw_volume[:, :, current_slice])

        return z_slices
    else:
        raise ValueError("Only x, y, or z axis is available")


def normalize_slice_values(scan_slices: list):
    """
    Normalizes slice image values for each axis to range 0-255 and dtype np.uint8

    :param scan_slices: list of slices for each axis
    :return: normalized list of slices for each axis
    """
    xyz_scan_slices = []
    for ax_scan_slice in scan_slices:
        data = ((ax_scan_slice.astype(np.float32)) / np.max(ax_scan_slice))
        data = data * 255
        data = data.astype(np.uint8)
        xyz_scan_slices.append(data)

    return xyz_scan_slices


def save_image_to_png(image: np.ndarray, save_path: Path):
    """
    Saves image to .png format on given path

    :param image: image to save
    :param save_path: path where image will be saved
    :return:
    """
    try:
        img = Image.fromarray(image)
        img.save(save_path, format="PNG", optimize=False, compress_level=0)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Create path {save_path.parent}")
