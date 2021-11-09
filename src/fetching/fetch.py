import io
import numpy as np
import geopandas as gpd
import os
from rasterio import transform
import requests

import rasterio
from imageio import imread

from zipfile import ZipFile
from rasterio.io import MemoryFile
from io import BytesIO

from .landtrendr import get_landtrendr_download_url, read_gee_url


def dem_from_tnm(bbox, width, height, inSR=3857, **kwargs):
    """
    Retrieves a Digital Elevation Model (DEM) image from The National Map (TNM)
    web service.

    Parameters
    ----------
    bbox : list-like
      list of bounding box coordinates (minx, miny, maxx, maxy)
    res : numeric
      spatial resolution to use for returned DEM (grid cell size)
    inSR : int
      spatial reference for bounding box, such as an EPSG code (e.g., 4326)
    Returns
    -------
    dem : numpy array
      DEM image as array
    """
    BASE_URL = ''.join([
        'https://elevation.nationalmap.gov/arcgis/rest/',
        'services/3DEPElevation/ImageServer/exportImage?'
    ])

    params = dict(bbox=','.join([str(x) for x in bbox]),
                  bboxSR=inSR,
                  size=f'{width},{height}',
                  imageSR=inSR,
                  time=None,
                  format='tiff',
                  pixelType='F32',
                  noData=None,
                  noDataInterpretation='esriNoDataMatchAny',
                  interpolation='+RSP_BilinearInterpolation',
                  compression=None,
                  compressionQuality=None,
                  bandIds=None,
                  mosaicRule=None,
                  renderingRule=None,
                  f='image')
    for key, value in kwargs.items():
        params.update({key: value})

    r = requests.get(BASE_URL, params=params)
    dem = imread(io.BytesIO(r.content))

    return dem


def get_landtrendr_from_gee(bbox, year, epsg):
    url = get_landtrendr_download_url(bbox, year, epsg)
    ras, profile = read_gee_url(url)
    return ras, profile
