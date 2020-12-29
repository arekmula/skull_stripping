from argparse import ArgumentParser
from pathlib import Path

from segmentation.dataset import split_first_dataset, split_second_dataset


def main(args):

    first_dataset_path = args.first_dataset_path
    second_dataset_path = args.second_dataset_path

    train_set, val_set = split_first_dataset(Path(first_dataset_path))
    if second_dataset_path is not None:
        second_train_set, second_val_set = split_second_dataset(Path(second_dataset_path))
        train_set += second_train_set
        val_set += second_val_set

    print(len(train_set))
    print(len(val_set))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--first_dataset_path", metavar="first_dataset_path", type=str, required=True)
    parser.add_argument("--second_dataset_path", metavar="second_dataset_path", type=str, default=None)

    args, _ = parser.parse_known_args()

    main(args)
