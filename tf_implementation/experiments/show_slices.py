from argparse import ArgumentParser
from matplotlib import pyplot as plt
from pathlib import Path

from segmentation.utils import load_raw_volume, load_labels_volume, show_slices, allow_memory_growth


def main(args):
    slice_raw_volume_path = args.slice_raw_volume_path
    slice_label_path = args.slice_label_path

    # affine variable will be needed when saving a prediction
    raw_volume, affine = load_raw_volume(Path(slice_raw_volume_path))
    mask_volume = load_labels_volume(Path(slice_label_path))

    show_slices([raw_volume[raw_volume.shape[0] // 2],  # Middle 2D slice in x axis
                 raw_volume[:, raw_volume.shape[1] // 2],  # Middle 2D slice in y axis
                 raw_volume[:, :, raw_volume.shape[2] // 2]])  # Middle 2D slice in z axis

    show_slices([mask_volume[mask_volume.shape[0] // 2],  # Middle 2D slice in x axis
                 mask_volume[:, mask_volume.shape[1] // 2],  # Middle 2D slice in y axis
                 mask_volume[:, :, mask_volume.shape[2] // 2]])  # Middle 2D slice in z axis

    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()

    # pass arguments
    parser.add_argument('--allow-memory-growth', action='store_true', default=True)
    parser.add_argument('-srvp', '--slice_raw_volume_path', metavar="slice_raw_volume_path", type=str,
                        help="Path to the raw volume of slice you want to show")

    parser.add_argument('-slp', '--slice_label_path', metavar="slice_label_path", type=str,
                        help="Path to the label of slice you want to show")

    args, _ = parser.parse_known_args()

    if args.allow_memory_growth:
        allow_memory_growth()

    main(args)
