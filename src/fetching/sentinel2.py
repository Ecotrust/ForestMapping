'''
Created on Nov 3, 2021

@author: sloreno
'''

import os
import glob

import rasterio
import ee
import numpy as np
import geopandas as gpd

import requests
from io import BytesIO
from zipfile import ZipFile
from concurrent.futures import as_completed, ThreadPoolExecutor
from tqdm.notebook import tqdm


#create sentinel2 rasters using earth engine
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

    BANDS = ['B4','B3','B2','B8','B11','B12','B5','B6','B7','B8A']
    MAP_BANDS_TO = ['R','G','B','NIR','SWIR1','SWIR2','RE1','RE2','RE3','RE4']

    return leafoff_coll.select(BANDS, MAP_BANDS_TO), leafon_coll.select(BANDS, MAP_BANDS_TO)

def get_medoid(collection, bands=['R','G','B','NIR','SWIR1', 'SWIR2', 'BRIGHTNESS', 'GREENNESS', 'WETNESS', 'NDVI', 'SAVI', 'ENDVI']):
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
    bands = ['R','G','B','NIR','SWIR1', 'SWIR2']
    leafoff_img = get_medoid(leafoff_coll, bands)
    leafon_img = get_medoid(leafon_coll, bands)           

    return leafoff_img, leafon_img

def get_sentinel2(path_to_tiles, year, filepath):
    
    gdf = gpd.read_file(path_to_tiles)
    print('Preparing download URLs for {:,d} tiles'.format(len(gdf)))
    OUT_ROOT = filepath
    to_download = []
    cell_ids = gdf['CELL_ID'].astype(str).values


    print('\n', year, flush=True)
    out_dir = os.path.join(OUT_ROOT, 'sentinel', str(year))
    already_done = [os.path.basename(x) for x in glob.glob(f'{out_dir}/*') if os.path.basename(x).split('_')[0] in cell_ids]
    print('Already have {:,.0f} tiles for {}'.format(len(already_done)/2, year))

    for idx, row in gdf.iterrows():
        cell_id = str(row['CELL_ID'])
        leafon_out = f'{cell_id}_sentinel-leaf-on.tif'
        leafoff_out = f'{cell_id}_sentinel-leaf-off.tif'
        # get the tile bounding box for filtering landsat images
        bbox_tile = row['geometry'].bounds
        xmin, ymin, xmax, ymax = bbox_tile
        (xmin, ymin) = np.floor((xmin, ymin))
        (xmax, ymax) = np.ceil((xmax, ymax))
        aoi = ee.Geometry.Rectangle((xmin, ymin, xmax, ymax), 
                                        proj=f'EPSG:{gdf.crs.to_epsg()}', 
                                        evenOdd=True, 
                                        geodesic=False)

        # get a composite leaf-on images
        leafoff_img, leafon_img = get_sentinel2_composites(aoi, year)

        for outfile, img in zip([leafon_out, leafoff_out], [leafon_img, leafoff_img]):
            url_params = dict(name=outfile.split('.')[0],
                                  filePerBand=False,
                                  scale=30,
                                  crs=f'EPSG:{gdf.crs.to_epsg()}',
                                  formatOptions={'cloudOptimized':True})
            url = img.clip(aoi).getDownloadURL(url_params)
            to_download.append((url, outfile, out_dir))

        # report progress
        if idx % 100 == 0 and idx > 0:
            print()
        if (idx+1) % 10 == 0:
            print('{:,d}'.format(idx+1), end='')
        else:
            print('.', end='')

    return to_download

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
