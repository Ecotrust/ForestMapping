import numpy as np


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
