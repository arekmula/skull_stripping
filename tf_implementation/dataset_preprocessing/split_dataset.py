from argparse import ArgumentParser
from pathlib import Path

from segmentation.dataset import split_first_dataset, split_second_dataset, save_scan_to_xyz_slices


def main(args):

    first_dataset_path = args.first_dataset_path
    second_dataset_path = args.second_dataset_path

    train_set, val_set = split_first_dataset(Path(first_dataset_path))
    if second_dataset_path is not None:
        second_train_set, second_val_set = split_second_dataset(Path(second_dataset_path))
        train_set += second_train_set
        val_set += second_val_set

    train_set_path = args.train_set_save_path

    scan_number = 0

    for scan in train_set:
        save_scan_to_xyz_slices(scan, train_set_path, scan_number)
        scan_number += 1

    val_set_path = args.val_set_save_path
    for scan in val_set:
        save_scan_to_xyz_slices(scan, val_set_path, scan_number)
        scan_number += 1


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--first_dataset_path", metavar="first_dataset_path", type=str, required=True)
    parser.add_argument("--second_dataset_path", metavar="second_dataset_path", type=str, default=None)
    parser.add_argument("--train_set_save_path", metavar="train_set_save_path", type=str, required=True)
    parser.add_argument("--val_set_save_path", metavar="val_set_save_path", type=str, required=True)

    args, _ = parser.parse_known_args()

    main(args)
