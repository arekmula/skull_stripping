from argparse import ArgumentParser
from pathlib import Path
from matplotlib import pyplot as plt

from segmentation.dataset import scans_generator


def main(args):
    train_path = Path(args.train_dir_path)
    val_path = Path(args.val_dir_path)

    train_scans, val_scans, train_samples, val_samples = scans_generator(train_path, val_path)
    # Get batch of data
    images, masks = next(train_scans)

    for i in range(len(images)):
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        print("Please note that preprocessing for specific backbone might cause strange image for human eye")
        plt.imshow(images[i], cmap="gray")

        fig.add_subplot(1, 2, 2)
        plt.imshow(masks[i], cmap="gray")

        plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_dir_path", metavar="train_dir_path", type=str, required=True)
    parser.add_argument("--val_dir_path", metavar="val_dir_path", type=str, required=True)

    args, _ = parser.parse_known_args()

    main(args)
