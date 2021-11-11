import requests
from zipfile import ZipFile
from rasterio.io import MemoryFile
from io import BytesIO

import ee
from geetools import composite


def harmonize_to_oli(image):
    """Applies linear adjustments to transform earlier sensors to more closely
    match LANDSAT 8 OLI as described in:

        Roy et al. (2016). "Characterization of Landsat-7 to Landsat-8
        reflective wavelength and normalized difference vegetation index
        continuity." Remote Sensing of Environment (185): 57–70.
        https://doi.org/10.1016/j.rse.2015.12.024
    """

    ROY_COEFS = { # B, G, R, NIR, SWIR1, SWIR2
        'intercepts': ee.Image.constant(
            [0.0003, 0.0088, 0.0061, 0.0412, 0.0254, 0.0172]
            ).multiply(10000),  # this scales LS7ETM to match LS8OLI scaling
        'slopes': ee.Image.constant(
            [0.8474, 0.8483, 0.9047, 0.8462, 0.8937, 0.9071]
            )
        }

    harmonized = image.select(['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2'])\
                 .multiply(ROY_COEFS['slopes'])\
                 .add(ROY_COEFS['intercepts'])\
                 .round()\
                 .toShort()

    return harmonized

def mask_stuff(image):
    """Masks pixels likely to be cloud, shadow, water, or snow in a LANDSAT
    image based on the `pixel_qa` band."""
    qa = image.select('pixel_qa')

    shadow = qa.bitwiseAnd(8).eq(0)
    snow = qa.bitwiseAnd(16).eq(0)
    cloud = qa.bitwiseAnd(32).eq(0)
    water = qa.bitwiseAnd(4).eq(0)

    masked = image.updateMask(shadow).updateMask(cloud).updateMask(snow).updateMask(water)

    return masked

def get_landsat_collection(aoi, start_year, end_year, band='SWIR1'):
    """Builds a time series of summertime LANDSAT imagery within an Area of
    Interest, returning a single composite image for a single band each year.
    """
    years = range(start_year, end_year+1)
    images = []

    for i, year in enumerate(years):
        if year >= 1984 and year <= 2011:
            sensor, bands = 'LT05', ['B1', 'B2', 'B3', 'B4', 'B5', 'B7']
        elif year == 2012:
            continue
        elif year >= 2013:
            sensor, bands = 'LC08', ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']

        landsat = ee.ImageCollection(f'LANDSAT/{sensor}/C01/T1_SR')

        coll = landsat.filterBounds(aoi)\
              .filterDate(f'{year}-06-15', f'{year}-09-15')
        masked = coll.map(mask_stuff)\
                .select(bands, ['B','G','R','NIR','SWIR1','SWIR2'])
        medoid = composite.medoid(masked, discard_zeros=True)

        if sensor != 'LC08':
            img = harmonize_to_oli(medoid)
        else:
            img = medoid

        if band == 'NBR':
            nbr = img.normalizedDifference(['NIR', 'SWIR2'])\
                     .rename('NBR').multiply(1000)
            img = img.addBands(nbr)

        images.append(img.select([band])\
                      .set('system:time_start',
                           coll.first().get('system:time_start')))

    return ee.ImageCollection(images)

def parse_landtrendr_result(lt_result, current_year,
                            flip_disturbance=False, big_fast=False, sieve=False):
    """Parses a LandTrendr segmentation result, returning an image that
    identifies the years since the largest disturbance.

    Parameters
    ----------
    lt_result : image
      result of running ee.Algorithms.TemporalSegmentation.LandTrendr on an
      image collection
    current_year : int
       used to calculate years since disturbance
    flip_disturbance: bool
      whether to flip the sign of the change in spectral change so that
      disturbances are indicated by increasing reflectance
    big_fast : bool
      consider only big and fast disturbances
    sieve : bool
      filter out disturbances that did not affect more than 11 connected pixels
      in the year of disturbance

    Returns
    -------
    img : image
      an image with four bands:
        ysd - years since largest spectral change detected
        mag - magnitude of the change
        dur - duration of the change
        rate - rate of change
    """
    lt = lt_result.select('LandTrendr')
    is_vertex = lt.arraySlice(0, 3, 4)  # 'Is Vertex' row - yes(1)/no(0)
    verts = lt.arrayMask(is_vertex)  # vertices as boolean mask

    left, right = verts.arraySlice(1, 0, -1), verts.arraySlice(1, 1, None)
    start_yr, end_yr = left.arraySlice(0, 0, 1), right.arraySlice(0, 0, 1)
    start_val, end_val = left.arraySlice(0, 2, 3),  right.arraySlice(0, 2, 3)

    ysd = start_yr.subtract(current_year-1).multiply(-1)  # time since vertex
    dur = end_yr.subtract(start_yr)  # duration of change
    if flip_disturbance:
        mag = end_val.subtract(start_val).multiply(-1)  # magnitude of change
    else:
        mag = end_val.subtract(start_val)

    rate = mag.divide(dur)  # rate of change

    # combine segments in the timeseries
    seg_info = ee.Image.cat([ysd, mag, dur, rate])\
               .toArray(0)\
               .mask(is_vertex.mask())

    # sort by magnitude of disturbance
    sort_by_this = seg_info.arraySlice(0,1,2).toArray(0)
    seg_info_sorted = seg_info.arraySort(sort_by_this.multiply(-1))  # flip to sort in descending order
    biggest_loss = seg_info_sorted.arraySlice(1, 0, 1)

    img = ee.Image.cat(
              biggest_loss.arraySlice(0,0,1).arrayProject([1]).arrayFlatten([['ysd']]),
              biggest_loss.arraySlice(0,1,2).arrayProject([1]).arrayFlatten([['mag']]),
              biggest_loss.arraySlice(0,2,3).arrayProject([1]).arrayFlatten([['dur']]),
              biggest_loss.arraySlice(0,3,4).arrayProject([1]).arrayFlatten([['rate']])
              )

    if big_fast:
        # get disturbances larger than 100 and less than 4 years in duration
        dist_mask = img.select(['mag']).gt(100).And(img.select(['dur']).lt(4))

        img = img.mask(dist_mask)

    if sieve:
        MAX_SIZE =  128  #  maximum map unit size in pixels
        # group adjacent pixels with disturbance in same year
        # create a mask identifying clumps larger than 11 pixels
        mmu_patches = img.int16()\
                      .select(['ysd'])\
                      .connectedPixelCount(MAX_SIZE, True)\
                      .gte(11)

        img = img.updateMask(mmu_patches)

    return img.round().toShort()

def get_landtrendr_download_url(bbox, year, epsg, scale=30):
    xmin, ymin, xmax, ymax = bbox
    aoi = ee.Geometry.Rectangle((xmin, ymin, xmax, ymax),
                                proj=f'EPSG:{epsg}',
                                evenOdd=True,
                                geodesic=False)

    swir_coll = get_landsat_collection(aoi, 1984, year, band='SWIR1')
    nbr_coll = get_landsat_collection(aoi, 1984, year, band='NBR')

    LT_PARAMS = {
      'maxSegments': 6,
      'spikeThreshold': 0.9,
      'vertexCountOvershoot': 3,
      'preventOneYearRecovery': True,
      'recoveryThreshold': 0.25,
      'pvalThreshold': 0.05,
      'bestModelProportion': 0.75,
      'minObservationsNeeded': 6
    }

    swir_result = ee.Algorithms.TemporalSegmentation.LandTrendr(swir_coll, **LT_PARAMS)
    nbr_result = ee.Algorithms.TemporalSegmentation.LandTrendr(nbr_coll, **LT_PARAMS)

    swir_img = parse_landtrendr_result(swir_result, year)\
                  .set('system:time_start',
                      swir_coll.first().get('system:time_start'))
    nbr_img = parse_landtrendr_result(nbr_result, year, flip_disturbance=True)\
                .set('system:time_start',
                      nbr_coll.first().get('system:time_start'))

    lt_img = ee.Image.cat(
        swir_img.select(['ysd'], ['ysd_swir1']),
        swir_img.select(['mag'], ['mag_swir1']),
        swir_img.select(['dur'], ['dur_swir1']),
        swir_img.select(['rate'], ['rate_swir1']),
        nbr_img.select(['ysd'], ['ysd_nbr']),
        nbr_img.select(['mag'], ['mag_nbr']),
        nbr_img.select(['dur'], ['dur_nbr']),
        nbr_img.select(['rate'], ['rate_nbr']),
      ).set('system:time_start',
            swir_img.get('system:time_start'))

    url_params = dict(filePerBand=False,
                      scale=scale,
                      crs=f'EPSG:{epsg}',
                      formatOptions={'cloudOptimized':True})
    url = lt_img.clip(aoi).getDownloadURL(url_params)

    return url

def read_gee_url(url):
    """Given a download URL generated by Google Earth Engine, downloads the
    and opens the raster into memory.

    Parameters
    ----------
    url : str
      URL from which the raster will be downloaded, generated by Google Earth
      Engine.

    Returns
    -------
    ras : arr
      raster read into an array
    profile : dict
      raster metadata
    """
    response = requests.get(url)
    data = ZipFile(BytesIO(response.content)).read('download.tif')
    with MemoryFile(data) as memfile:
        with memfile.open() as src:
            ras = src.read()
            profile = src.profile
    return ras, profile


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
    MEDOID_BANDS = ['B','G','R','NIR','SWIR1', 'SWIR2']
    leafoff_img = get_medoid(leafoff_coll, MEDOID_BANDS)
    leafon_img = get_medoid(leafon_coll, MEDOID_BANDS)

    img = ee.Image.cat(
        leafoff_img.select(['B'], ['B_LEAFOFF']),
        leafoff_img.select(['G'], ['G_LEAFOFF']),
        leafoff_img.select(['R'], ['R_LEAFOFF']),
        leafoff_img.select(['RE1']), ['RE1_LEAFOFF']),
        leafoff_img.select(['RE2']), ['RE2_LEAFOFF']),
        leafoff_img.select(['RE3']), ['RE3_LEAFOFF']),
        leafoff_img.select(['RE4']), ['RE4_LEAFOFF']),
        leafoff_img.select(['NIR'], ['NIR_LEAFOFF']),
        leafoff_img.select(['SWIR1'], ['SWIR1_LEAFOFF']),
        leafoff_img.select(['SWIR2'], ['SWIR2_LEAFOFF']),
        leafon_img.select(['B'], ['B_LEAFON']),
        leafon_img.select(['G'], ['G_LEAFON']),
        leafon_img.select(['R'], ['R_LEAFON']),
        leafon_img.select(['RE1']), ['RE1_LEAFON']),
        leafon_img.select(['RE2']), ['RE2_LEAFON']),
        leafon_img.select(['RE3']), ['RE3_LEAFON']),
        leafon_img.select(['RE4']), ['RE4_LEAFON']),
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
