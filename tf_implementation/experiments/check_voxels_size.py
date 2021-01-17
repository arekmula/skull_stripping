from argparse import ArgumentParser
from pathlib import Path

from segmentation.utils import print_voxels_size


def main(args):

    first_dataset_path = args.first_dataset_path
    second_dataset_path = args.second_dataset_path

    print_voxels_size(Path(first_dataset_path))
    if second_dataset_path is not None:
        print_voxels_size(Path(second_dataset_path))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--first_dataset_path", metavar="first_dataset_path", type=str, required=True)
    parser.add_argument("--second_dataset_path", metavar="second_dataset_path", type=str, default=None)

    args, _ = parser.parse_known_args()

    main(args)
