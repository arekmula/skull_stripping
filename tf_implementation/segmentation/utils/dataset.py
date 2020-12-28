import nibabel as nib
import numpy as np

from matplotlib import pyplot as plt
from pathlib import Path
from typing import Tuple, List


def load_raw_volume(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data: nib.Nifti1Image = nib.load(str(path))
    data = nib.as_closest_canonical(data)
    raw_data = data.get_fdata(caching='unchanged', dtype=np.float32)
    return raw_data, data.affine


def load_labels_volume(path: Path) -> np.ndarray:
    return load_raw_volume(path)[0].astype(np.uint8)


def save_labels(data: np.ndarray, affine: np.ndarray, path: Path):
    nib.save(nib.Nifti1Image(data, affine), str(path))


def show_slices(slices: List[np.ndarray]):
    fig, axes = plt.subplots(1, len(slices))
    for i, data_slice in enumerate(slices):
        axes[i].imshow(data_slice.T, cmap="gray", origin="lower")
