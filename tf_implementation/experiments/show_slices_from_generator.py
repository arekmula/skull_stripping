from argparse import ArgumentParser
from pathlib import Path
from matplotlib import pyplot as plt

from segmentation.dataset import scans_generator


def main(args):
    path = Path(args.path)

    scans = scans_generator(path)

    # Get batch of data
    images, masks = next(scans)

    for i in range(len(images)):
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.imshow(images[i], cmap="gray")

        fig.add_subplot(1, 2, 2)
        plt.imshow(masks[i], cmap="gray")

        plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", metavar="path", type=str, required=True)

    args, _ = parser.parse_known_args()

    main(args)
