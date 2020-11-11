import argparse
import logging
import math
import ntpath
import os
import sys
from glob import glob

import PIL
import click
import numpy as np
from PIL import Image, ImageStat
from tqdm import tqdm

DATASET_DIR = "dataset"
IMG_DIR = "images"
MSK_DIR = "masks"
MAPS_EXT = "tif"
GT_MAPS_EXT = "png"
SET_RESOLUTION = (512, 512)

# We are working with pretty large images.
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def def_args(parent_parser=None):
    """
    Defines argument parser.
    """
    if parent_parser:
        child_parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    else:
        child_parser = argparse.ArgumentParser(
            description="Creates a dataset from orthophotos and ground truth images. "
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
    child_parser.add_argument(
        "-s",
        dest="stat",
        action="store_false",
        help="Disable statistics gathering for the dataset.",
    )
    child_parser.add_argument(
        "-1",
        dest="skip_black",
        action="store_false",
        help="Export images containing black pixels. I.e out of map-bounds.",
    )
    child_parser.add_argument(
        "-2",
        dest="skip_no_class",
        action="store_false",
        help="Export images containing only background class.",
    )
    child_parser.add_argument(
        "-3",
        dest="skip_all_water",
        action="store_false",
        help="Export images containing only water bodies.",
    )

    return child_parser


def get_dataset(path: str):
    """
    Gets the dataset as two lists of paths to the files.
    Dataset must have the following convention:
    dataset/
        images/
            1_x.png
        masks/
            1_y.png

    Parameters
    ----------
    path : path to the dataset

    Returns
    -------

    """
    image_path = os.path.join(path, "images")
    mask_path = os.path.join(path, "masks")
    image_files = []
    mask_files = []
    if os.path.isdir(image_path) and os.path.isdir(mask_path):
        for (dir_path, _, filenames) in os.walk(image_path):
            for f in filenames:
                image_files.append(os.path.join(dir_path, f))
        for (dir_path, _, filenames) in os.walk(mask_path):
            for f in filenames:
                mask_files.append(os.path.join(dir_path, f))
        x = len(image_files)
        y = len(mask_files)
        if x != y:
            logger.warning(
                "Found un-even numbers of x-y for dataset. x = %i, y = %i.", x, y
            )
        if x == 0:
            logger.warning("Found 0 existing sets.")
            return image_files, mask_files
        logger.info("Found %i sets in existing dataset.", x)
        return image_files, mask_files

    logger.error("Could not locate x and y folder.")
    sys.exit()


def img_to_set(img_msk):
    """
    Returns image or mask with corresponding set-pair.
    Parameters
    ----------
    img_msk : str
        image or mask path

    Returns
    -------
    tup : tuple
        (img_file_path, msk_file_path)
    string : str
        "img_file_path, msk_file_path"

    """
    tup = ("", "")
    string = ""
    if "_x" in img_msk:
        tup = (img_msk, img_msk.replace("_x", "_y").replace("images", "masks"))
        string = "{},{}".format(tup[0], tup[1])
    if "_y" in img_msk:
        tup = (img_msk, img_msk.replace("_y", "_x").replace("masks", "images"))
        string = "{},{}".format(tup[1], tup[0])

    return tup, string


def file_to_list(path: str):
    """
    Gets images and mask paths from file.
    Parameters
    ----------
    path : str
        File path

    Returns
    -------
    files : list[(img,msk)]

    """
    files = []
    with open(path, "r") as file_handler:
        for line in file_handler.readlines():
            split = line.split(",")
            files.append((split[1], split[2].replace("\n", "")))

    return files


def contains_black(image):
    """
    Returns true if image contains black pixels.
    Parameters
    ----------
    image : PIL.Image

    Returns
    -------
    bool

    """
    extrema = ImageStat.Stat(image).extrema
    r = extrema[0][0]
    g = extrema[1][0]
    b = extrema[2][0]
    if r == 0 and g == 0 and b == 0:
        return True
    return False


def no_classes(mask):
    """
    Returns true if mask is all black.
    Parameters
    ----------
    mask : PIL.Image

    Returns
    -------
    bool

    """
    extrema = ImageStat.Stat(mask).extrema
    r = extrema[0][1]
    g = extrema[1][1]
    b = extrema[2][1]
    if r == 0 and g == 0 and b == 0:
        return True
    return False


def only_water_bodies(image):
    """
    Returns true if image contains black pixels.
    Parameters
    ----------
    image : PIL.Image

    Returns
    -------
    bool

    """
    extrema = ImageStat.Stat(image).extrema
    r_l, r_h = extrema[0]
    g_l, g_h = extrema[1]
    b_l, b_h = extrema[2]
    if r_l == g_l == b_l == r_h == g_h == b_h == 127:
        return True
    return False


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
        Tuple that contains path to dataset, x-files and y-files.

    """
    ds_path = os.path.join(os.path.join(path, DATASET_DIR))
    x_path = os.path.join(ds_path, IMG_DIR)
    y_path = os.path.join(ds_path, MSK_DIR)
    paths = [ds_path, x_path, y_path]
    for p in paths:
        try:
            os.mkdir(p)
            logger.info("Directory %s created", ds_path)
        except Exception as e:
            logger.warning("Failed to create directories: %s", e)
    return ds_path, x_path, y_path


def gen_crop_area(x_res, y_res, dim):
    """
    Generates boxes for crop function.

    Parameters
    ----------
    x_res : int
        resolution height
    y_res : int
        resolution width
    dim : tuple
        Map size

    Returns
    -------
    crop_area : list
        List of boxes as tuples

    """
    crop_area = []
    for x in range(math.floor(dim[0] / x_res)):
        for y in range(math.floor(dim[1] / y_res)):
            left = x * x_res
            right = left + x_res
            upper = y * y_res
            lower = upper + y_res
            crop_area.append((left, upper, right, lower))

    return crop_area


def reduce_and_grayscale(mask):
    """
    Synthetically reduces interpolation to nearest neighbour and converts to grayscale.
    Parameters
    ----------
    mask : PIL.Image
        Mask to reduce anc convert.

    Returns
    -------
    gray_mask : PIL:Image
        Mask as grayscale, and with only three different colours.

    """
    r, _, _, _ = Image.Image.split(mask)

    r = np.asarray(r)

    water = np.logical_and(r <= 190, r > 63)
    buildings = r > 190

    np_mask = np.zeros_like(r)
    np_mask[water] = 127
    np_mask[buildings] = 255

    gray_mask = Image.fromarray(np_mask)
    return gray_mask


def get_gt_map(raster_map, gt_maps):
    """
    Returns the corresponding ground truth map for a certain map.
    Parameters
    ----------
    raster_map : str
        Map to get GT from.
    gt_maps : list
        List of all GT maps.

    Returns
    -------
    m : str
        Path to GT map.

    """
    for gt_m in gt_maps:
        map_name = ntpath.basename(raster_map).split(".")[0]
        gt_map_name = ntpath.basename(gt_m).split(".")[0].replace("_y", "")
        if map_name == gt_map_name:
            logger.info("X: %s Y: %s", map_name, gt_map_name)
            return gt_m

    logger.warning("Unable to get ground truth image for %s", raster_map)
    return None


def get_maps(path: str, ext: str) -> list:
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


def check_existing_dataset(path: str):
    """
    Checks if dataset exists and returns number of samples if it does.

    Parameters
    ----------
    path

    Returns
    -------

    """
    x_path = os.path.join(path, IMG_DIR)
    y_path = os.path.join(path, MSK_DIR)
    if os.path.isdir(x_path) and os.path.isdir(y_path):
        _, _, x_files = next(os.walk(x_path))
        _, _, y_files = next(os.walk(y_path))
        x = len(x_files)
        y = len(y_files)
        if x != y:
            logger.warning(
                "Found un-even numbers of x-y for dataset. x = %i, y = %i.", x, y
            )
            return -1
        if x == 0:
            logger.info("Found 0 existing sets.")
            return 0
        logger.info("Found %s sets in existing dataset.", x)
        return x
    logger.error("Could not locate x and y folder.")
    sys.exit()


def add_to_set(
    map_crop: PIL.Image,
    gt_map_crop: PIL.Image,
    skip_black: bool = True,
    skip_water: bool = True,
    skip_no_class: bool = True,
) -> bool:
    """
    Tests the set, return if it should be added or not.
    Parameters
    ----------
    map_crop : PIL.Image
        Cropped image from map
    gt_map_crop : PIL.Image
        Cropped image from ground truth map
    skip_black : bool
        Skip images containing black pixels
    skip_water : bool
        Skip images containing only water bodies
    skip_no_class : bool
        Skip images containing only background class

    Returns
    -------
    bool

    """
    if skip_black and contains_black(map_crop):
        return False
    if skip_water and only_water_bodies(gt_map_crop):
        return False
    if skip_no_class and no_classes(gt_map_crop):
        return False
    return True


def create_sets(
    path: tuple,
    maps_ath: str,
    gt_maps_path: str,
    ds_index: int = 0,
    skip_black: bool = True,
    skip_water: bool = True,
    skip_no_class: bool = True,
):
    """
    Creates the sets en each folder x and y.
    Parameters
    ----------tuple
    dirs : tuple
        Tuple of three strings: dataset path, x files path and y files path.
    index : int
        Index to start sets from.
    maps_dir : str
        Path to maps
    gt_maps_dir : str
        Path to GT maps
    skip_black : bool
        Skips images containing black pixels. I.e out of raster map bounds.
    skip_water : bool
        Skips images containing only water bodies.
    skip_no_class : bool
        Skips images containing only background class.

    """
    maps = get_maps(maps_ath, MAPS_EXT)
    gt_maps = get_maps(gt_maps_path, GT_MAPS_EXT)
    logger.info(
        "Found %i aerial maps and %i ground truth maps.", len(maps), len(gt_maps)
    )
    with tqdm(total=len(maps), desc="Maps") as pbar:
        for m in maps:
            try:
                ortho_map = Image.open(m)
                gt_map = Image.open(get_gt_map(m, gt_maps))
                if ortho_map.size == gt_map.size:
                    boxes = gen_crop_area(
                        SET_RESOLUTION[0], SET_RESOLUTION[1], ortho_map.size
                    )
                    with tqdm(
                        total=len(boxes),
                        leave=False,
                        desc="Sets for {}".format(os.path.basename(m)),
                    ) as pbar2:
                        for b in boxes:
                            map_crop = ortho_map.crop(b)
                            gt_map_crop = gt_map.crop(b)
                            if add_to_set(
                                map_crop,
                                gt_map_crop,
                                skip_black=skip_black,
                                skip_water=skip_water,
                                skip_no_class=skip_no_class,
                            ):
                                map_fn = os.path.join(
                                    path[1], "{}_x.png".format(ds_index)
                                )
                                gt_map_fn = os.path.join(
                                    path[2], "{}_y.png".format(ds_index)
                                )
                                map_crop.save(map_fn)
                                gray_gt_map_crop = reduce_and_grayscale(gt_map_crop)
                                gray_gt_map_crop.save(gt_map_fn)
                                ds_index += 1

                            pbar2.set_description(
                                "Sets for {}(index: {})".format(
                                    os.path.basename(m), ds_index
                                )
                            )
                            pbar2.update()
                else:
                    continue
            except Exception as e:
                logger.error("Error occurred while creating set: %s", e)
                logger.error("Skipping %s", m)
            pbar.update()


def get_statistics(path):
    """
    Gathers statistics such as mean, standard deviation and class percentages.
    Prints the statistics and writes them to a file at path.

    Parameters
    ----------
    path : str
        Dataset path

    """
    images, masks = get_dataset(path)
    buildings = 0
    background = 0
    water = 0
    mean = np.zeros(3)
    std = np.zeros(3)

    with tqdm(
        total=len(images), desc="Getting statistics..", leave=False, position=0
    ) as pbar:
        for i, m in zip(images, masks):
            image = Image.open(i)
            stat = ImageStat.Stat(image)
            mean = np.add(np.asarray(stat.mean), mean)
            std = np.add(np.asarray(stat.stddev), std)

            mask = Image.open(m)
            for c in mask.getcolors():
                if c[1] == 0:
                    background += c[0]
                if c[1] == 127:
                    water += c[0]
                if c[1] == 255:
                    buildings += c[0]
            pbar.update()

    mean = np.divide(mean, len(images))
    std = np.divide(std, len(images))

    all_pixels = buildings + background + water
    buildings_perc = (buildings / all_pixels) * 100
    water_perc = (water / all_pixels) * 100
    background_perc = (background / all_pixels) * 100

    filename = os.path.join(path, "myfile.txt")

    with open(filename, "w") as file:
        file.write("Mean: {}\n".format(mean))
        file.write("Standard deviation: {}\n".format(std))

        file.write("Building pixels: {}\n".format(buildings))
        file.write("Water pixels: {}\n".format(water))
        file.write("Background pixels: {}\n".format(background))
        file.write("Building percentage: {}\n".format(buildings_perc))
        file.write("Water pixels: {}\n".format(water_perc))
        file.write("Background pixels: {}\n".format(background_perc))

    with open(filename, "r") as file_r:
        print(file_r.read())


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

    create_sets(dirs, maps_dir, gt_maps_dir, index)

    if args.stat:
        get_statistics(os.path.join(args.outdir, "dataset"))
