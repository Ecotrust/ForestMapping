'''
Created on Nov 3, 2021

@author: sloreno
'''
import os
import glob

import numpy as np
import geopandas as gpd
import rasterio
import ee
from geetools import composite

import requests
from io import BytesIO
from zipfile import ZipFile
from concurrent.futures import as_completed, ThreadPoolExecutor
from tqdm.notebook import tqdm

def harmonize_to_oli(image):
    """Applies linear adjustments to transform earlier sensors to more closely
    match LANDSAT 8 OLI as described in:
    
        Roy et al. (2016). "Characterization of Landsat-7 to Landsat-8 
        reflective wavelength and normalized difference vegetation index 
        continuity." Remote Sensing of Environment (185): 57–70. 
        https://doi.org/10.1016/j.rse.2015.12.024
    """

    ROY_COEFS = { # re-ordered to be R, G, B, NIR, SWIR1, SWIR2
        'intercepts': ee.Image.constant(
            [0.0061, 0.0088, 0.0003, 0.0412, 0.0254, 0.0172]
            ).multiply(10000),  # this scales LS7ETM to match LS8OLI scaling
        'slopes': ee.Image.constant(
            [0.9047, 0.8483, 0.8474, 0.8462, 0.8937, 0.9071]
            )
        }
        
    harmonized = image.select(['R', 'G', 'B', 'NIR', 'SWIR1', 'SWIR2'])\
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
            sensor, bands = 'LT05', ['B3', 'B2', 'B1', 'B4', 'B5', 'B7']
        elif year == 2012:
            continue
        elif year >= 2013:
            sensor, bands = 'LC08', ['B4', 'B3', 'B2', 'B5', 'B6', 'B7']
            
        landsat = ee.ImageCollection(f'LANDSAT/{sensor}/C01/T1_SR')
    
        coll = landsat.filterBounds(aoi)\
              .filterDate(f'{year}-06-15', f'{year}-09-15')
        masked = coll.map(mask_stuff)\
                .select(bands, ['R','G','B','NIR','SWIR1','SWIR2'])
        medoid = composite.medoid(masked, discard_zeros=True)

        if sensor != 'LC08':
            img = harmonize_to_oli(medoid)
        else:
            img = medoid

        if band == 'NBR':
            nbr = img.normalizedDifference(['NIR', 'SWIR2']).rename('NBR').multiply(1000)
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

def get_landtrendr_download_params(params):
    geom, cell_id, year, epsg, filepath = params
    OUT_ROOT = filepath
    out_dir = os.path.join(OUT_ROOT, 'landtrendr', str(year))
    outfile = f'{cell_id}_landtrendr.tif'

    bbox_tile = geom.bounds
    xmin, ymin, xmax, ymax = bbox_tile
    (xmin, ymin) = np.floor((xmin, ymin))
    (xmax, ymax) = np.ceil((xmax, ymax))
    aoi = ee.Geometry.Rectangle((xmin,ymin,xmax,ymax),
                                proj=f'EPSG:{epsg}', 
                                evenOdd=True, 
                                geodesic=False)
                
    swir_collection = get_landsat_collection(aoi, 1984, year, band='SWIR1')
    nbr_collection = get_landsat_collection(aoi, 1984, year, band='NBR')
    
    lt_swir_result = ee.Algorithms.TemporalSegmentation.LandTrendr(
                    swir_collection, **LT_PARAMS)
    lt_nbr_result = ee.Algorithms.TemporalSegmentation.LandTrendr(
        nbr_collection, **LT_PARAMS)
    
    lt_swir_img = parse_landtrendr_result(lt_swir_result, year)\
                  .set('system:time_start', 
                      swir_collection.first().get('system:time_start'))
    lt_nbr_img = parse_landtrendr_result(lt_nbr_result, year, flip_disturbance=True)\
                .set('system:time_start', 
                      nbr_collection.first().get('system:time_start'))
                
    lt_img = ee.Image.cat(
        lt_swir_img.select(['ysd'], ['ysd_swir1']),
        lt_swir_img.select(['mag'], ['mag_swir1']),
        lt_swir_img.select(['dur'], ['dur_swir1']),
        lt_swir_img.select(['rate'], ['rate_swir1']),
        lt_nbr_img.select(['ysd'], ['ysd_nbr']),
        lt_nbr_img.select(['mag'], ['mag_nbr']),
        lt_nbr_img.select(['dur'], ['dur_nbr']),
        lt_nbr_img.select(['rate'], ['rate_nbr']),
      ).set('system:time_start', 
            lt_swir_img.get('system:time_start'))

    url_params = dict(name=outfile.split('.')[0],
                      filePerBand=False,
                      scale=30,
                      crs=f'EPSG:{epsg}',
                      formatOptions={'cloudOptimized':True})
    url = lt_img.clip(aoi).getDownloadURL(url_params)

    return url, outfile, out_dir

def fetch_unzip_from_url(url, filename, out_dir, check_valid=True, retry=True):
    """Fetches a zipfile from a URL and extracts the specified file 
    from the zip archive to out_dir.

    This is primarily intended to download a zipped GeoTiff.
    """
    out_path = os.path.join(out_dir, filename)
    
    if not os.path.exists(out_path):
        response = requests.get(url)
        try:
            zip = ZipFile(BytesIO(response.content))
        except: # downloaded zip is corrupt/failed
            return None
        out_path = zip.extract(filename, out_dir)    

    if check_valid:
        try:
            with rasterio.open(out_path) as src:
                ras = src.read(masked=True)
        except:
            print(f'Failed to fetch {filename}.')
            os.remove(out_path)

            if retry:
                return fetch_unzip_from_url(url, filename, out_dir, retry=False)
            else:
                return None

    return out_path

def multithreaded_download_preparer(path_to_tiles, year, filename, num_threads=12, overwrite=False):
    gdf = gpd.read_file(path_to_tiles)
    to_download = []
    epsg = gdf.crs.to_epsg()
    
    OUT_ROOT = '/content/drive/Shared drives/stand_mapping/data/interim/training_tiles/'
    cell_ids = gdf['CELL_ID'].astype(str).values
    

    out_dir = os.path.join(OUT_ROOT, 'landtrendr')
    already_done = [os.path.basename(x).split('_')[0] for x in glob.glob(f'{out_dir}/*') if os.path.basename(x).split('_')[0] in cell_ids]
    print('Already have {:,d} tiles for {}.'.format(len(already_done), year), end=' ')
    to_do = gdf.loc[~gdf.CELL_ID.isin(already_done)]
        
    if len(to_do) > 0:
        with ThreadPoolExecutor(12) as executor:
            print('Preparing download URLs from Google Earth Engine.')
            jobs = [executor.submit(get_landtrendr_download_params, 
                                    (row.geometry, row.CELL_ID, year, epsg, filename)) for idx, row in to_do.iterrows()]        
            for job in tqdm(as_completed(jobs), total=len(jobs), desc=str(year)):
                to_download.append(job.result())
    else:
        print()
    return to_download

def multithreaded_download(to_download, num_threads=12):
    if len(to_download) > 0:
        with ThreadPoolExecutor(12) as executor:
            print('Starting to download files from Google Earth Engine.')
            jobs = [executor.submit(fetch_unzip_from_url, *params) for params in to_download]
            results = []
            
            for job in tqdm(as_completed(jobs), total=len(jobs)):
                results.append(job.result())
        return results
    else:
        return


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
