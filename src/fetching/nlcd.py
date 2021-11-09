import numpy as np
import requests
import io
from imageio import imread


def nlcd_from_mrlc(bbox, width, height, layer, inSR=4326, nlcd=True, **kwargs):
    """
    Retrieves National Land Cover Data (NLCD) Layers from the Multiresolution
    Land Characteristics Consortium's web service.

    Parameters
    ----------
    bbox : list-like
      list of bounding box coordinates (minx, miny, maxx, maxy)
    width, height : int
      width and height (in pixels) of image to be returned
    layer : str
      title of layer to retrieve (e.g., 'NLCD_2001_Land_Cover_L48')
    inSR : int
      spatial reference for bounding box, such as an EPSG code (e.g., 4326)
    nlcd : bool
      if True, will re-map the values returned to the NLCD land cover codes

    Returns
    -------
    img : numpy array
      map image as array
    """
    BASE_URL = ''.join([
        'https://www.mrlc.gov/geoserver/mrlc_display/wms?',
        'service=WMS&request=GetMap',
    ])

    params = dict(bbox=','.join([str(x) for x in bbox]),
                  crs=f'epsg:{inSR}',
                  width=width,
                  height=height,
                  format='image/tiff',
                  layers=layer)
    for key, value in kwargs.items():
        params.update({key: value})

    r = requests.get(BASE_URL, params=params)
    img = imread(io.BytesIO(r.content), format='tiff')

    if nlcd:
        MAPPING = {
            1: 11,  # open water
            2: 12,  # perennial ice/snow
            3: 21,  # developed, open space
            4: 22,  # developed, low intensity
            5: 23,  # developed, medium intensity
            6: 24,  # developed, high intensity
            7: 31,  # barren land (rock/stand/clay)
            8: 32,  # unconsolidated shore
            9: 41,  # deciduous forest
            10: 42,  # evergreen forest
            11: 43,  # mixed forest
            12: 51,  # dwarf scrub (AK only)
            13: 52,  # shrub/scrub
            14: 71,  # grasslands/herbaceous,
            15: 72,  # sedge/herbaceous (AK only)
            16: 73,  # lichens (AK only)
            17: 74,  # moss (AK only)
            18: 81,  # pasture/hay
            19: 82,  # cultivated crops
            20: 90,  # woody wetlands
            21: 95,  # emergent herbaceous wetlands
        }

        k = np.array(list(MAPPING.keys()))
        v = np.array(list(MAPPING.values()))

        mapping_ar = np.zeros(k.max() + 1, dtype=v.dtype)
        mapping_ar[k] = v
        img = mapping_ar[img]

    return img

def colorize_nlcd(img):
    """Assigns colors to an NLCD land cover image.

    Parameters
    ----------
    img : arr, shape (H, W)
      array containing NLCD land cover classifications

    Returns
    -------
    land_color : arr, shape (H, W, 3)
      RGB image of land cover types
    """
    COLOR_MAP = {
        0: [0, 0, 0],  # nodata
        11: [84, 117, 168],  # open water
        12: [255, 255, 255],  # perennial ice and snow
        21: [232, 209, 209],  # developed, open space
        22: [226, 158, 140],  # developed, low intensity
        23: [255, 0, 0],  # developed, medium intensity
        24: [181, 0, 0],  # developed, high intensity
        31: [210, 205, 192],  # barren land (rock/sand/clay)
        41: [133, 199, 126],  # deciduous forest
        42: [56, 129, 78],  # evergreen forest
        43: [212, 231, 176],  # mixed forest
        51: [175, 150, 60],  # dwarf scrub
        52: [220, 202, 143],  # shrub/scrub
        71: [253, 233, 170],  # grassland/herbaceous
        72: [209, 209, 130],  # sedge/herbaceous
        73: [163, 204, 81],  # lichens
        74: [130, 186, 158],  # moss
        81: [251, 246, 93],  # pasture/hay
        82: [202, 145, 70],  # cultivated crops
        90: [200, 230, 248],  # woody wetlands
        95: [100, 179, 213],  # emergent herbaceous wetlands
    }
    land_color = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for cov in np.unique(img):
        mask = img == cov
        land_color[mask] = COLOR_MAP[cov]

    return land_color
