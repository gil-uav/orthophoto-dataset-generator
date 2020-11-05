import os
import shutil

from ordage import make_dirs, get_maps


def test_check_existing_dataset():
    assert False


def test_make_dirs():
    make_dirs(".")
    ds_path = os.path.join(os.path.join(".", "dataset"))
    x_path = os.path.join(ds_path, "x")
    y_path = os.path.join(ds_path, "y")
    assert os.path.isdir(x_path)
    assert os.path.isdir(y_path)
    shutil.rmtree(ds_path)


def test_get_all_files():
    assert False


def test_get_statistics():
    assert False


def test_get_maps():
    test = get_maps("/media/vegovs/MoseSchrute/Rasterkart")
    assert False


def test_get_gt():
    assert False
