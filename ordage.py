import logging
import math
import ntpath
import os
import sys
from glob import glob
import click

from PIL import Image

import argparse

from tqdm import tqdm

DATASET_DIR = "dataset"
X_DIR = "x"
Y_DIR = "y"
MAPS_EXT = "tif"
GT_MAPS_EXT = "png"
SET_RESOLUTION = (512, 512)

# We are working with pretty large images.
Image.MAX_IMAGE_PIXELS = None

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def def_args(parent_parser=None):
    if parent_parser:
        child_parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    else:
        child_parser = argparse.ArgumentParser(
            description="Creates a dataset from ortophotos and ground truth images. "
            'Ortophotos must end with "_x", and ground truths with "_y"'
        )
    child_parser.add_argument(
        "-o",
        dest="outdir",
        metavar="PATH",
        type=str,
        required=True,
        default=".",
        help="Root of dataset folder. Appends samples if dataset exists.",
    )
    child_parser.add_argument(
        "-x",
        dest="x_dir",
        metavar="PATH",
        type=str,
        required=True,
        help="Directory where ortophotos are located.",
    )
    child_parser.add_argument(
        "-y",
        dest="y_dir",
        metavar="PATH",
        type=str,
        required=True,
        help="Directory where ground truth labels are located.",
    )

    return child_parser


def make_dirs(path):
    """
    Creates directories for the dataset.
    Parameters
    ----------
    path : str
        Where to place the dataset.

    Returns
    -------
    dirs : tuple
        Tuple that containts path to dataset, x-files and y-files.

    """
    ds_path = os.path.join(os.path.join(path, DATASET_DIR))
    x_path = os.path.join(ds_path, X_DIR)
    y_path = os.path.join(ds_path, Y_DIR)
    paths = [ds_path, x_path, y_path]
    for p in paths:
        try:
            os.mkdir(p)
            log.info("Directory {} created".format(ds_path))
        except Exception as e:
            log.warning("Failed to create directories: {}".format(e))
    return (ds_path, x_path, y_path)


def check_existing_dataset(path: str):
    """
    Checks if dataset exists and returns number of samples if it does.

    Parameters
    ----------
    path

    Returns
    -------

    """
    x_path = os.path.join(path, "x")
    y_path = os.path.join(path, "y")
    if os.path.isdir(x_path) and os.path.isdir(y_path):
        _, _, x_files = next(os.walk(x_path))
        _, _, y_files = next(os.walk(y_path))
        x = len(x_files)
        y = len(y_files)
        if x != y:
            log.warning(
                "Found un-even numbers of x-y for dataset. x = {}, y = {}.".format(x, y)
            )
            return -1
        if x == 0:
            log.info("Found 0 existing sets.")
            return 0
        else:
            log.info("Found {} sets in existing dataset.".format(x))
            return x
    else:
        log.error("Could not locate x and y folder.")
        sys.exit()


def create_sets(dirs: tuple, index: int, maps_dir: str, gt_maps_dir: str):
    """
    Creates the sets en each folder x and y.
    Parameters
    ----------
    dirs : tuple
        Tuple of three strings: dataset path, x files path and y files path.
    index : int
        Index to start sets from.
    maps_dir :
        Path to maps
    gt_maps_dir
        Path to GT maps

    """
    maps = get_maps(maps_dir, MAPS_EXT)
    gt_maps = get_maps(gt_maps_dir, GT_MAPS_EXT)
    log.info(
        "Found {} orthomaps and {} ground truth maps.".format(len(maps), len(gt_maps))
    )
    index = index
    with tqdm(total=len(maps), desc="Maps") as pbar:
        for m in maps:
            try:
                orthomap = Image.open(m)
                gt_map = Image.open(get_gt_map(m, gt_maps))
                if orthomap.size == gt_map.size:
                    boxes = gen_crop_area(
                        SET_RESOLUTION[0], SET_RESOLUTION[1], orthomap.size
                    )
                    with tqdm(
                        total=len(boxes),
                        leave=False,
                        desc="Sets for {}".format(os.path.basename(m)),
                    ) as pbar2:
                        for b in boxes:
                            map_crop = orthomap.crop(b)
                            gt_map_crop = gt_map.crop(b)
                            map_fn = os.path.join(dirs[1], "{}_x.png".format(index))
                            gt_map_fn = os.path.join(dirs[2], "{}_y.png".format(index))
                            map_crop.save(map_fn)
                            gt_map_crop.save(gt_map_fn)
                            pbar2.set_description(
                                "Sets for {}(index: {})".format(
                                    os.path.basename(m), index
                                )
                            )
                            pbar2.update()
                            index += 1
                else:
                    continue
            except Exception as e:
                log.error("Error occurred while creating set: {}".format(e))
                log.error("Skipping {}".format(m))
            pbar.update()


def gen_crop_area(x_res, y_res, size):
    """
    Genereates boxes for crop function.

    Parameters
    ----------
    x_res : int
        resolution height
    y_res : int
        resolution width
    size : int
        Map size

    Returns
    -------
    crop_area : list
        List of boxes as tuples

    """
    crop_area = []
    for x in range(math.floor(size[0] / x_res)):
        for y in range(math.floor(size[1] / y_res)):
            left = x * x_res
            right = left + x_res
            upper = y * y_res
            lower = upper + y_res
            crop_area.append((left, upper, right, lower))

    return crop_area


def get_gt_map(map, gt_maps):
    """
    Returns the corresponding ground truth map for a certain map.
    Parameters
    ----------
    map : str
        Map to get GT from.
    gt_maps : list
        List of all GT maps.

    Returns
    -------
    m : str
        Path to GT map.

    """
    for m in gt_maps:
        map_fn = ntpath.basename(map).split(".".format(MAPS_EXT))[0]
        gt_map_fn = (
            ntpath.basename(m).split(".".format(GT_MAPS_EXT))[0].replace("_y", "")
        )
        if map_fn == gt_map_fn:
            log.info("X: {} Y: {}".format(map_fn, gt_map_fn))
            return m


def get_maps(path: str, ext: str):
    """
    Returns all map-files from path.
    Parameters
    ----------
    path : str
        Where map-files are.
    ext : str
        .tif, .png etc.

    Returns
    -------
    result : list
        List of paths to each file.

    """
    result = [
        y for x in os.walk(path) for y in glob(os.path.join(x[0], "*.{}".format(ext)))
    ]
    return result


if __name__ == "__main__":
    parser = def_args()
    args = parser.parse_args()

    dirs = make_dirs(args.outdir)
    index = check_existing_dataset(dirs[0])
    if index < 0:
        if not click.confirm("Do you want to overwrite dataset?", default=True):
            sys.exit()
        index = 0
    elif index > 0:
        if not click.confirm(
            "Do you want to continue from index {}?\n Starts from 0 if no.".format(
                index
            ),
            default=True,
        ):
            index = 0

    maps_dir = args.x_dir
    gt_maps_dir = args.y_dir

    create_sets(dirs, index, maps_dir, gt_maps_dir)
