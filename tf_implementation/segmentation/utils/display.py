import nibabel as nib
import numpy as np

from matplotlib import pyplot as plt
from pathlib import Path
from typing import List


def show_slices(slices: List[np.ndarray]):
    """
    Shows slices from axis x, y, z

    :param slices: list of image slices from axis x, y, z
    :return:
    """
    fig, axes = plt.subplots(1, len(slices))
    for i, data_slice in enumerate(slices):
        axes[i].imshow(data_slice.T, cmap="gray", origin="lower")


def print_voxels_size(path: Path):
    """
    Prints size of voxels in millimeters

    :param path: path to folder containing masks
    :return:
    """
    for scan_path in path.iterdir():
        if scan_path.name.endswith('mask.nii.gz'):
            print(nib.load(str(scan_path)).header.get_zooms())
