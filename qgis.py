import os

from qgis.core import QgsProject
from glob import glob

project = QgsProject.instance()
root = QgsProject.instance().layerTreeRoot()


def import_map(path_to_tif: str, root):
    """
    Imports a .tif file as map-layer.

    Parameters
    ----------
    path_to_tif : str
        Path to .tif file.
    root
        Project instance layer root.

    Returns
    -------

    """
    rlayer = QgsRasterLayer(
        path_to_tif, os.path.basename(path_to_tif).replace(".tif", "")
    )
    if not rlayer.isValid():
        print("Layer failed to load!")
    iface.addRasterLayer(path_to_tif, os.path.basename(path_to_tif).replace(".tif", ""))


def import_all_maps(path):
    """
    Imports all .tif files to project as layers.
    Parameters
    ----------
    path : str
        Path to folder containing .tif files(Ortophotos)

    Returns
    -------

    """
    maps = [y for x in os.walk(path) for y in glob(os.path.join(x[0], "*.tif"))]
    for m in ops:
        import_map(m, root)
        print("{} imported.".format(os.path.basename(m)))


def get_ortophoto_layers():
    """
    Returns all ortophoto layers from project.
    Returns
    -------
    layers : list
        List of ortophoto layers.

    """
    layers = [
        l
        for l in QgsProject.instance().mapLayers().values()
        if l.name().startswith("33")
    ]
    layers = [
        l
        for l in QgsProject.instance().mapLayers().values()
        if l.type() == QgsMapLayer.RasterLayer
    ]
    return layers


def export_basedata_as_img(layer, export_path: str):
    """
    Saves ground-truth for layer as .png file.

    Parameters
    ----------
    export_path : str
        Where to save the image.
    layer
        Orthophoto layer to produce ground-truth map from.

    """
    outfile = os.path.join(export_path, "{}_y.png".format(layer.name()))

    settings = QgsMapSettings()
    settings.setLayers(
        [
            QgsProject.instance().mapLayersByName("fkb_bygning_omrade")[0],
            QgsProject.instance().mapLayersByName("fkb_vann_omrade")[0],
        ]
    )
    settings.setBackgroundColor(QColor(0, 0, 0))
    settings.setOutputSize(QSize(layer.width(), layer.height()))
    settings.setExtent(layer.extent())
    render = QgsMapRendererParallelJob(settings)

    def finished():
        img = render.renderedImage()
        img.save(outfile, "png")

    render.finished.connect(finished)
    render.start()
    print("Ground truth image export of {} started.".format(layer.name()))
    from qgis.PyQt.QtCore import QEventLoop

    loop = QEventLoop()
    render.finished.connect(loop.quit)
    loop.exec_()
    print("Ground truth image of {} exported to: {}".format(layer.name(), outfile))


def export_all_ground_truth_maps(export_path: str):
    """
    Exports ground truth images of all orthophoto layers.

    export_path : str
        Where to save the image.
    """
    for l in get_ortophoto_layers():
        export_basedata_as_img(l, export_path)

