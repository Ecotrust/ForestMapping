import ee
from .landtrendr import read_gee_url


def maskS2clouds(img):
    qa = img.select('QA60')

    # bits 10 and 11 are clouds and cirrus
    cloudBitMask = ee.Number(2).pow(10).int()
    cirrusBitMask = ee.Number(2).pow(11).int()

    # both flags set to zero indicates clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(\
            qa.bitwiseAnd(cirrusBitMask).eq(0))
    return img.updateMask(mask).addBands(img.metadata('system:time_start'))

def maskS2Edges(img):
    return img.updateMask(
        img.select('B8A').mask().updateMask(img.select('B9').mask()))


def get_sentinel2_collections(aoi, year):
    """Returns a SENTINEL-2 collection filtered to a specific area of
    interest and timeframe and with clouds masked."""

    leafoff_start_date, leafoff_end_date = f'{year-1}-10-01', f'{year}-03-31'
    leafon_start_date, leafon_end_date = f'{year}-04-01', f'{year}-09-30'

    # Filter input collections by desired data range and region.
    s2Sr = ee.ImageCollection("COPERNICUS/S2_SR")

    leafoff_coll = s2Sr.filterBounds(aoi).filterDate(leafoff_start_date, leafoff_end_date)
    leafoff_coll = leafoff_coll.map(maskS2clouds).map(maskS2Edges)

    leafon_coll = s2Sr.filterBounds(aoi).filterDate(leafon_start_date, leafon_end_date)
    leafon_coll = leafon_coll.map(maskS2clouds).map(maskS2Edges)

    BANDS = ['B2','B3','B4','B8','B11','B12','B5','B6','B7','B8A']
    MAP_TO = ['B','G','R','NIR','SWIR1','SWIR2','RE1','RE2','RE3','RE4']

    return leafoff_coll.select(BANDS, MAP_TO), leafon_coll.select(BANDS, MAP_TO)

def get_medoid(collection, bands=['B','G','R','NIR','SWIR1', 'SWIR2']):
    """Makes a medoid composite of images in an image collection.

    Adapted to Python from a Javascript version here:
    https://github.com/google/earthengine-community/blob/73178fa9e0fd370783f871fe73eb38912f4c8bb9/toolkits/landcover/impl/composites.js#L88
    """
    median = collection.select(bands).median()  # per-band median across collection

    def med_diff(image):
        """Calculates squared difference of each pixel from median of each band.
        This functions is nested in `get_medoid` because it uses the median of
        the collection containing the image.
        """
        distance = image.select(bands).spectralDistance(median, 'sed')\
                   .multiply(-1.0)\
                   .rename('medoid_distance')

        return image.addBands(distance)

    indexed = collection.map(med_diff)

    # qualityMosaic selects pixels for a mosaic that have the highest value
    # in the user-specified band
    mosaic = indexed.qualityMosaic('medoid_distance')
    band_names = mosaic.bandNames().remove('medoid_distance')

    return mosaic.select(band_names)


def get_sentinel2_composites(aoi, year):
    """Returns median composite images for leaf-on and leaf-off timeperiods from
     SENTINEL-2 for an area of interest for the specified year.
    """

    leafoff_coll, leafon_coll = get_sentinel2_collections(aoi, year)

    # get a median composite image
    bands = ['B','G','R','NIR','SWIR1', 'SWIR2']
    leafoff_img = get_medoid(leafoff_coll, bands)
    leafon_img = get_medoid(leafon_coll, bands)
    # leafoff_img = composite.medoid(leafoff_coll, bands=bands, discard_zeros=True)
    # leafon_img = composite.medoid(leafon_coll, bands=bands, discard_zeros=True)

    img = ee.Image.cat(
        leafoff_img.select(['B'], ['B_LEAFOFF']),
        leafoff_img.select(['G'], ['G_LEAFOFF']),
        leafoff_img.select(['R'], ['R_LEAFOFF']),
        leafoff_img.select(['NIR'], ['NIR_LEAFOFF']),
        leafoff_img.select(['SWIR1'], ['SWIR1_LEAFOFF']),
        leafoff_img.select(['SWIR2'], ['SWIR2_LEAFOFF']),
        leafon_img.select(['B'], ['B_LEAFON']),
        leafon_img.select(['G'], ['G_LEAFON']),
        leafon_img.select(['R'], ['R_LEAFON']),
        leafon_img.select(['NIR'], ['NIR_LEAFON']),
        leafon_img.select(['SWIR1'], ['SWIR1_LEAFON']),
        leafon_img.select(['SWIR2'], ['SWIR2_LEAFON']),
      ).set('system:time_start',
            leafon_img.get('system:time_start'))

    return img

def get_sentinel2_download_url(bbox, year, epsg, scale=10):
    """Returns URL from which SENTINEL-2 composite image can be downloaded."""
    xmin, ymin, xmax, ymax = bbox
    aoi = ee.Geometry.Rectangle((xmin, ymin, xmax, ymax),
                                proj=f'EPSG:{epsg}',
                                evenOdd=True,
                                geodesic=False)

    img = get_sentinel2_composites(aoi, year)
    url_params = dict(filePerBand=False,
                      scale=scale,
                      crs=f'EPSG:{epsg}',
                      formatOptions={'cloudOptimized':True})
    url = img.clip(aoi).getDownloadURL(url_params)

    return url

def s2_from_gee(bbox, year, epsg, scale=10):
    """Fetches an 12-band raster generated from Google Earth Engine containing
    a leaf-off and leaf-on composite image for the user-specified area of
    interest and year.

    The 12-bands in the returned raster are:
        1. Blue (LEAFOFF)
        2. Green (LEAFOFF)
        3. Red (LEAFOFF)
        4. NIR (LEAFOFF)
        5. SWIR1 (LEAFOFF)
        6. SWIR2 (LEAFOFF)
        7. Blue (LEAFON)
        8. Green (LEAFON)
        9. Red (LEAFON)
        10. NIR (LEAFON)
        11. SWIR1 (LEAFON)
        12. SWIR2 (LEAFON)

    Parameters
    ----------
    bbox : list-like
      (xmin, ymin, xmax, ymax) coordinates for bounding box
    year : int
      year for leaf-on imagery (leaf-off will reach back into previous year)
    epsg : int
      EPSG code used to define projection
    """
    url = get_sentinel2_download_url(bbox, year, epsg, scale)
    ras, profile = read_gee_url(url)

    return ras, profile
