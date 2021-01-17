from argparse import ArgumentParser
from pathlib import Path
from matplotlib import pyplot as plt

from segmentation.dataset import split_first_dataset, split_second_dataset, load_raw_volume, load_labels_volume,\
    get_axes_slices_from_volume, normalize_slice_values
from segmentation.utils import show_slices


def main(args):

    first_dataset_path = args.first_dataset_path
    second_dataset_path = args.second_dataset_path

    train_set, val_set = split_first_dataset(Path(first_dataset_path))
    if second_dataset_path is not None:
        second_train_set, second_val_set = split_second_dataset(Path(second_dataset_path))
        train_set += second_train_set
        val_set += second_val_set

    for scan in train_set:
        raw_volume, affine = load_raw_volume(scan[0])
        mask_volume = load_labels_volume(scan[1])

        xyz_scan_slices = get_axes_slices_from_volume(raw_volume=raw_volume)
        xyz_scan_slices = normalize_slice_values(scan_slices=xyz_scan_slices)

        xyz_labels_slices = get_axes_slices_from_volume(raw_volume=mask_volume)

        show_slices(xyz_scan_slices)
        show_slices(xyz_labels_slices)

        plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--first_dataset_path", metavar="first_dataset_path", type=str, required=True)
    parser.add_argument("--second_dataset_path", metavar="second_dataset_path", type=str, default=None)

    args, _ = parser.parse_known_args()

    main(args)
