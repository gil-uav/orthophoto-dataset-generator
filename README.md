# Orthopoto dataset generator
**Members** : <a href="https://github.com/vegovs">Vegard Bergsvik Øvstegård</a>

**Supervisors** : <a href="https://www.mn.uio.no/ifi/personer/vit/jimtoer/">Jim Tørresen</a>

## Description

This repository contains some scripts and tools to aid in the automation of dataset generation.
The dataset is meant to contain [orthopohots](https://www.sciencedirect.com/topics/earth-and-planetary-sciences/orthophoto) with
segmented buildings as ground truth.

## Dependencies
* [Python](https://www.python.org/) (version 3.6, 3.7 or 3.8)
* [Pip](https://virtualenv.pypa.io/en/latest/)
* [virtualenv](https://virtualenv.pypa.io/en/latest/) or:
* [qgis](https://qgis.org/en/site/)

## Installation

```console
git clone https://github.com/gil-uav/orthophoto-dataset-generator
```

#### virtualenv

```console
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Qgis usage
To use functions from [qgis.py](https://github.com/gil-uav/orthophoto-dataset-generator/blob/master/qgis.py), open
script file in Qgis and run the functions at the bottom of the script. Ground truth(GT) vector maps must be imported manually.
Changes style(colors) etc as you see fit. NB! Change naming of the GT layers in the `export_basedata_as_img` function:
```python
def export_basedata_as_img(layer, export_path: str):
    ...
    settings.setLayers(
        [
            QgsProject.instance().mapLayersByName("gtlayer1")[0],
            QgsProject.instance().mapLayersByName("grlayer2")[0],
        ]
    )
    ...
```


To import all othophoto maps from a folder(works recursively):
```python
#script stuff..#
import_all_maps("/path/to/maps")
```

To export all ground truth maps from orthophoto maps:
```python
#script stuff..#
export_all_ground_truth_maps("/path/to/export/folder")
```

## Generator usage
```
usage: ordage.py [-h] -o PATH -x PATH -y PATH [-1] [-2] [-3]

Creates a dataset from orthophotos and ground truth images. Ortophotos must
end with "_x", and ground truths with "_y"

optional arguments:
  -h, --help  show this help message and exit
  -o PATH     Root of dataset folder. Appends samples if dataset exists.
  -x PATH     Directory where ortophotos are located.
  -y PATH     Directory where ground truth labels are located.
  -1          Export images containing black pixels. I.e out of map-bounds.
  -2          Export images containing only background class.
  -3          Export images containing only water bodies.
```
