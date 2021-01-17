from argparse import ArgumentParser
from pathlib import Path

from segmentation.dataset import save_test_scan_to_xyz_slices


def main(args):

    first_dataset_path = Path(args.first_dataset_path)
    second_dataset_path = Path(args.second_dataset_path)
    save_base_path = Path(args.save_base_path)
    print(first_dataset_path)

    for scan in first_dataset_path.iterdir():
        save_test_scan_to_xyz_slices(scan, save_base_path)

    for scan in second_dataset_path.iterdir():
        save_test_scan_to_xyz_slices(scan, save_base_path, dataset_number=2)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--first_dataset_path", metavar="first_dataset_path", type=str, required=True)
    parser.add_argument("--second_dataset_path", metavar="second_dataset_path", type=str, required=True)
    parser.add_argument("--save_base_path", metavar="save_base_path", type=str, required=True)

    args, _ = parser.parse_known_args()

    main(args)
