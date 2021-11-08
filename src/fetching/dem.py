'''
Created on Nov 2, 2021

@author: sloreno
'''


import io
import numpy as np
import geopandas as gpd
import os
from rasterio import transform
import requests

import rasterio
from skimage import filters
from imageio import imread


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
    
    #res = 10
    #width = int(abs(bbox[2] - bbox[0]) // res)
    #height = int(abs(bbox[3] - bbox[1]) // res)

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

def fetch_dems(path_to_tiles, out_dir, path_to_landtrendr, overwrite=False):
    gdf = gpd.read_file(path_to_tiles)
    epsg = gdf.crs.to_epsg()
    print('Fetching DEMs for {:,d} tiles'.format(len(gdf)))
    
    PROFILE = {
        'driver': 'GTiff',
        'interleave': 'band',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
        'compress': 'lzw',
        'nodata': -9999,
        'dtype': rasterio.int16,
        'count': 1,
        }

    ## loop through all the geometries in the geodataframe and fetch the DEM

    for idx, row in gdf.iterrows():
        xmin, ymin, xmax, ymax = row['geometry'].bounds
        xmin, ymin = np.floor((xmin, ymin))
        xmax, ymax = np.ceil((xmax, ymax))
        bbox = [xmin, ymin, xmax, ymax]

        #width, height = xmax-xmin, ymax-ymin
        #trf = transform.from_bounds(xmin, ymin, xmax, ymax, width, height)
        
        with rasterio.open(os.path.join(path_to_landtrendr, f'{row.CELL_ID}_landtrendr.tif')) as dest:
            out_meta = dest.meta

        ## don't bother fetching data if we already have processed this tile
        outname = f'{row.CELL_ID}_dem.tif'
        outfile = os.path.join(out_dir, 'dem', outname)        
        if os.path.exists(outfile) and not overwrite:
            if idx % 100 == 0:
                print()
            if idx % 10 == 0:
                print(idx, end='')
            else:
                print('.', end='')
            continue
            
        dem = dem_from_tnm(bbox, out_meta["width"], out_meta["height"], 
                         qq=True, res=10, inSR=epsg, noData=-9999)
        
        ## apply a smoothing filter to mitigate stitching/edge artifacts
        dem = filters.gaussian(dem, 3)

        
        ## write the data to disk
        width, height = out_meta["width"], out_meta["height"]
        trf = transform.from_bounds(xmin, ymin, xmax, ymax, width, height)
        
        PROFILE.update(width=width, height=height)

        with rasterio.open(outfile, 'w', 
                           **PROFILE, crs=epsg, transform=trf) as dst:
            dst.write(dem.astype(rasterio.int16), 1)
            dst.set_band_unit(1, 'meters')
            dst.set_band_description(1, 'DEM retrieved from The National Map')
        
        ## report progress
        if idx % 100 == 0:
            print()
        if idx % 10 == 0:
            print(idx, end='')
        else:
            print('.', end='')
        
        
        
        
        
