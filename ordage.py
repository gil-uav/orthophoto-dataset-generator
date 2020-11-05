import logging
import os
from glob import glob

from PIL import Image
import argparse
import numpy as np

logging.basicConfig(level=logging.DEBUG)


def def_args(parent_parser=None):
    if parent_parser:
        child_parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    else:
        child_parser = argparse.ArgumentParser(
            description="Creates a dataset from ortophotos and ground truth images. "
            'Ortophotos must end with "_x", and ground truths with "_y"'
        )
    child_parser.add_argument(
        "outdir",
        dest="outdir",
        metavar="PATH",
        type=str,
        required=True,
        default=".",
        help="Root of dataset folder. Appends samples if dataset exists.",
    )
    child_parser.add_argument(
        "-x-dir",
        dest="x_dir",
        metavar="PATH",
        type=str,
        required=True,
        help="Directory where ortophotos are located.",
    )
    child_parser.add_argument(
        "-y-dir",
        dest="y_dir",
        metavar="PATH",
        type=str,
        required=True,
        help="Directory where ground truth labels are located.",
    )
    child_parser.add_argument(
        "-x-dim",
        dest="x_dim",
        metavar="[dim]",
        type=int,
        default=512,
        required=True,
        help="Patch width dimension.",
    )
    child_parser.add_argument(
        "-y-dim",
        dest="y_dim",
        metavar="[dim]",
        type=int,
        default=child_parser,
        help="Patch height dimension(Optional)",
    )
    child_parser.add_argument(
        "-tsp",
        dest="tsp",
        metavar="[-tsp]",
        type=int,
        default=0,
        help="Test/Holdout set percentage(Optional). No holdout set by default.",
    )
    child_parser.add_argument(
        "-v", dest="v", action="store_true", help="Verbose.",
    )

    return child_parser


def get_all_files():
    pass


def make_dirs(path):
    ds_path = os.path.join(os.path.join(path, "dataset"))
    x_path = os.path.join(ds_path, "x")
    y_path = os.path.join(ds_path, "y")
    try:
        os.mkdir(ds_path)
        logging.debug("Directory {} created".format(ds_path))
        os.mkdir(x_path)
        logging.debug("Directory {} created".format(x_path))
        os.mkdir(y_path)
        logging.debug("Directory {} created".format(y_path))
    except Exception as e:
        logging.error("Failed to create directories: {}".format(e))


def get_statistics():
    pass


def get_no_samples():
    pass


def check_existing_dataset(path: str):
    """
    Checks if dataset exists and returns number of samples if it does.

    Parameters
    ----------
    path

    Returns
    -------

    """
    if not os.path.isdir(path):
        return 0
    x_path = os.path.join(path, "x")
    y_path = os.path.join(path, "y")
    if os.path.isdir(x_path) and os.path.isdir(y_path):
        x = len(
            [n for n in os.listdir(x_path) if os.path.isfile(n) and n.endswith(".png")]
        )
        y = len(
            [n for n in os.listdir(y_path) if os.path.isfile(n) and n.endswith(".png")]
        )
        if x != y or x == 0:
            return 0
        else:
            return x


def get_maps(path: str):
    result = [y for x in os.walk(path) for y in glob(os.path.join(x[0], "*.tif"))]
    return result


def get_gt(path: str):
    result = [y for x in os.walk(path) for y in glob(os.path.join(x[0], "*.tif"))]
    pass


if __name__ == "__main__":
    parser = def_args()
    args = parser.parse_args()
    print(args.x_dir)
