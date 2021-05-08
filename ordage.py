import logging
import math
import os
import pickle
import sys
from glob import glob

import click
import kornia
import ntpath
import numpy as np
import PIL
import torch
from PIL import Image, ImageStat
from pytorch_lightning.metrics import FBeta, Precision
from pytorch_lightning.utilities import argparse
from torchvision import transforms
from tqdm import tqdm

import cv2
from unet.unet_model import UNet

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
        child_parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
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


def pil_to_opencv(pil_image):
    """
    Convert a PIL.Image to opencv readable image.
    Parameters
    ----------
    pil_image : PIL.Image

    Returns
    -------
    open_cv_image : numpy.ndarray

    """
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    return open_cv_image


def opencv_to_pil(opencv_image):
    """
    Converts a opencv readable image to PIL.Image
    Parameters
    ----------
    opencv_image : numpy.ndarray

    Returns
    -------
    pil_image: PIL.Image

    """
    # Convert RGB to BGR
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(opencv_image)

    return pil_image


def extract_rotated_set(map_image, gt_map_image, center, theta, width, height):
    map_shape = (map_image.shape[1], map_image.shape[0])
    gt_map_shape = (gt_map_image.shape[1], gt_map_image.shape[0])

    if map_shape != gt_map_shape:
        logger.error("Map and ground truth not same shape")
        sys.exit()

    matrix = cv2.getRotationMatrix2D(center=center, angle=theta, scale=1)

    rotated_map_image = cv2.warpAffine(
        src=map_image, M=matrix, dsize=map_shape)
    rotated_gt_map_image = cv2.warpAffine(
        src=gt_map_image, M=matrix, dsize=map_shape)

    x = int(center[0] - width / 2)
    y = int(center[1] - height / 2)

    map_patch = rotated_map_image[y: y + height, x: x + width]
    gt_map_patch = rotated_gt_map_image[y: y + height, x: x + width]

    return map_patch, gt_map_patch


def gen_center_points(x_res, y_res, dim):
    """
    Return list of center points.

    Parameters
    ----------
    x_res : int
    y_res : int
    dim :

    Returns
    -------

    """
    center_points = []

    for x in range(math.floor(dim[0] / x_res)):
        for y in range(math.floor(dim[1] / y_res)):
            x = (x + 1) * x_res
            y = (y + 1) * y_res
            center_points.append((x, y))

    return center_points


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


def to_grayscale_nn_interpolation(path):
    """
    Converts an image to grayscale and synthetically
    reduces to nearest neighbour interpolation
    Parameters
    ----------
    path : str
        Path to image.

    """
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    new_basename = basename.replace(".", "_gray.")
    new_path = os.path.join(dirname, new_basename)
    img = Image.open(path)
    new_img = reduce_and_grayscale(img)
    new_img.save(new_path)


def to_binary(path):
    """
    Converts an image to 1-bit image.
    Parameters
    ----------
    path : str
        Path to image.

    """
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    new_basename = basename.replace(".", "_binary.")
    new_path = os.path.join(dirname, new_basename)
    thresh = 1
    fn = lambda x: 255 if x > thresh else 0
    img = Image.open(path)
    new_img = img.convert('L').point(fn, mode='1')
    new_img.save(new_path)


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
        "Found %i aerial maps and %i ground truth maps.", len(
            maps), len(gt_maps)
    )
    with tqdm(total=len(maps), desc="Maps") as pbar:
        for m in maps:
            try:
                ortho_map = Image.open(m)
                gt_map = Image.open(get_gt_map(m, gt_maps))

                if ortho_map.size == gt_map.size:
                    ortho_map_cv2 = pil_to_opencv(ortho_map)
                    gt_map_cv2 = pil_to_opencv(gt_map)
                    boxes = gen_crop_area(
                        SET_RESOLUTION[0], SET_RESOLUTION[1], ortho_map.size
                    )
                    center_points = gen_center_points(
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
                                gray_gt_map_crop = reduce_and_grayscale(
                                    gt_map_crop)
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
        file.write("Water percentage: {}\n".format(water_perc))
        file.write("Background percentage: {}\n".format(background_perc))

    with open(filename, "r") as file_r:
        print(file_r.read())


def scan_for_gt_outliers(dataset_path, model_path):
    """
    Runs a model through the dataset, and tries to detect sets where
    classes lack annotations.

    Parameters
    ----------
    path : str
        Dataset path

    """
    candicates = []
    images, masks = get_dataset(dataset_path)
    images.sort()
    masks.sort()
    setattr(argparse, "_gpus_arg_default", lambda x: 0)
    model = UNet.load_from_checkpoint(checkpoint_path=model_path)
    model.cuda()
    model.eval()

    f_beta = FBeta(num_classes=1, beta=0.5).to(torch.device("cuda", 0))
    precision = Precision(num_classes=1, is_multiclass=False).to(
        torch.device("cuda", 0)
    )

    pixels = 512 * 512
    with tqdm(
            total=len(images),
            desc="Extracting sets with erronous ground truths.",
            leave=False,
            position=0,
    ) as pbar:
        for i, m in zip(images, masks):
            img = Image.open(i)
            msk = Image.open(m)

            img_tensor = transforms.ToTensor()(img).cuda()
            msk_tensor = transforms.ToTensor()(msk)
            msk_tensor = torch.gt(torch.sigmoid(msk_tensor), 0.5).cuda()

            if len(msk_tensor.unique()) != 2:
                continue

            img_tensor = torch.unsqueeze(img_tensor, 0)
            msk_tensor = torch.unsqueeze(msk_tensor, 0)

            preds = model(img_tensor)
            preds = torch.gt(torch.sigmoid(preds), 0.5)
            probs_img = torch.clamp(
                kornia.enhance.add_weighted(
                    src1=img_tensor, alpha=1.0, src2=preds, beta=0.5, gamma=0.0,
                ),
                max=1.0,
            )
            cv = preds / msk_tensor
            fp = torch.sum(cv == float("inf")).item() / pixels
            # f1 = f_beta(preds, msk_tensor)
            # p = precision(preds, msk_tensor)

            if fp > 0.1:
                candicates.append((i, m))
                with open("candidates.pkl", "wb") as f:
                    pickle.dump(candicates, f)

            img.close()
            msk.close()

            pbar.update()


def manually_filter_candidates():
    with open("candidates.pkl", "rb") as f:
        candidates = pickle.load(f)
    dir = os.path.dirname(os.path.dirname(candidates[0][0]))
    ex_images = os.path.join(dir, "excluded/images")
    ex_masks = os.path.join(dir, "excluded/masks")
    try:
        os.mkdir(os.path.join(dir, "excluded"))
        os.mkdir(ex_images)
        os.mkdir(ex_masks)
    except FileExistsError:
        print("Folders exist.")

    for img_path, msk_path in candidates:
        img = cv2.imread(img_path)
        msk = cv2.imread(msk_path)
        overlay = cv2.addWeighted(img, 0.5, msk, 0.5, 0.5)
        cv2.imshow("overlay", overlay)
        k = cv2.waitKey(0)

        if k == 27:
            break
        elif k == ord("x"):
            mv_img = os.path.join(ex_images, os.path.basename(img_path))
            mv_msk = os.path.join(ex_masks, os.path.basename(msk_path))
            os.rename(img_path, mv_img)
            os.rename(msk_path, mv_msk)
            candidates.remove((img_path, msk_path))
            with open("candidates.pkl", "wb") as f:
                pickle.dump(candidates, f)

    cv2.destroyAllWindows()


def main():
    parser = def_args()
    args = parser.parse_args()

    dirs = make_dirs(args.outdir)
    index = check_existing_dataset(dirs[0])

    if index < 0:
        if not click.confirm("Do you want to overwrite dataset?", default=True):
            sys.exit()
        index = 0
    elif index > 0:
        if not click.confirm("Do you want to continue from index {}?\n Starts from 0 if "
                             "no.".format(index), default=True):
            index = 0

    maps_dir = args.x_dir
    gt_maps_dir = args.y_dir

    create_sets(dirs, maps_dir, gt_maps_dir, index)

    if args.stat:
        get_statistics(os.path.join(args.outdir, "dataset"))


if __name__ == "__main__":
    # main()
    ds_path = "/media/vegovs/MoseSchrute/dataset"
    model_path = "./epoch=94-step=60703.ckpt"
    # scan_for_gt_outliers(ds_path, model_path)
    manually_filter_candidates()
