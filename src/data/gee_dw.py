# %%
from pathlib import Path
import os
import sys
import glob
from datetime import datetime

import numpy as np
import geopandas as gpd
import ee

# root = [p for p in Path(__file__).parents if p.name == "mapping_forest_types"][0]
# sys.path.append(os.path.abspath(root / "src"))

from utils import create_directory_tree, multithreaded_execution
from datafactory.gee_utils import GEEImageLoader

# YEAR = 2016
# QQ_SHP = "oregon_quarter_quads_selected.geojson"
# WORKERS = 20
# DATASET = 'test'
# OUT_DIR = f'processed/{DATASET}'

# %%
def get_dworld(
    bbox,
    year,
    path,
    prefix=None,
    season="leafon",
    overwrite=False,
    epsg=4326,
    scale=10,
    progressbar=None,
):
    """
    Fetch Dynamic World image url from Google Earth Engine (GEE) using a bounding box.
    Catalog https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1
    Parameters
    ----------
    month : int
        Month of year (1-12)
    year : int
        Year (e.g. 2019)
    bbox : list
        Bounding box in the form [xmin, ymin, xmax, ymax].
    Returns
    -------
    url : str
        GEE generated URL from which the raster will be downloaded.
    metadata : dict
        Image metadata.
    """
    def countPixels(image, geometry):
        counts = image.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=geometry,
            scale=scale,
            maxPixels=1e13,
        )
        try:
            return counts.get('label').getInfo()
        except:
            return 0

    if season == "leafoff":
        start_date = f"{year - 1}-10-01"
        end_date = f"{year}-03-31"
    elif season == "leafon":
        start_date = f"{year}-04-01"
        end_date = f"{year}-09-30"
    else:
        raise ValueError(f"Invalid season: {season}")

    bbox = ee.Geometry.BBox(*bbox)
    collection = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")\
                   .filterDate(start_date, end_date)\
                   .filterBounds(bbox)\
                   .select('label')

    ts_start = datetime.timestamp(datetime.strptime(start_date, "%Y-%m-%d"))
    ts_end = datetime.timestamp(datetime.strptime(end_date, "%Y-%m-%d"))
    
    # Extract medoid from collection
    # Reclass probabilities to 0-1
    img_median = collection.median()
    
    def get_medoid(image):
        diff = ee.Image(image).subtract(img_median).pow(ee.Image.constant(2))
        return diff.reduce('sum').addBands(image)

    medoid = collection.map(get_medoid)\
                       .reduce(ee.Reducer.min(2))\
                       .select([1], ['label'])\
                       .eq(1)\
                       .clip(bbox)
    reclass = medoid.expression('b(0) == 1 ? 1 : 0')
    tree_pixs = countPixels(medoid.mask(reclass), bbox)
    all_pixs = countPixels(medoid, bbox)

    # Skip if forest pixels are less than 10% of total pixels
    # Handle division by zero error
    try:
        pct = tree_pixs / all_pixs
    except:
        pct = 0

    if pct >= 0.1: 
        image = GEEImageLoader(medoid)
        # Set image metadata and params
        # image.metadata_from_collection(collection)
        image.set_property("system:time_start", ts_start * 1000)
        image.set_property("system:time_end", ts_end * 1000)
        image.set_params("scale", scale)
        image.set_params("crs", f"EPSG:{epsg}")
        image.set_params("region", bbox)
        image.set_viz_params("min", 0)
        image.set_viz_params("max", 1)
        image.set_viz_params('palette', ['black', '397D49'])
        image.id = f"{prefix}DynamicWorld_{year}_{season}"

        # Download cog
        # out_path = path / image.id
        # out_path.mkdir(parents=True, exist_ok=True)

        # image.save_metadata(out_path)
        image.to_geotif(path, overwrite=False)
        # image.save_preview(out_path, overwrite=True)
    else:
        print(f"Tile {prefix.replace('_', '')} has less than 10% of forest pixels, skipping...")
        
        
# %%
if __name__ == "__main__":

    # Initialize the Earth Engine module.
    # Setup your Google Earth Engine API key before running this script.
    # Instructions available at https://cloud.google.com/sdk/docs/install#deb
    # %%
    # ee.Authenticate() # run once after installing gcloud api
    ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com")
    # %%
    # Load the USGS QQ shapefile for Oregon state.
    qq_shp = gpd.read_file(root / "data/interim/usgs_grid" / QQ_SHP)
    match_tiles = glob.glob(f'{root.as_posix()}/data/processed/{DATASET}/3dep/**/*.tif', recursive=True)
    match_tiles = [int(Path(tile).stem.split('_')[0]) for tile in match_tiles] 
    qq_shp = qq_shp[qq_shp.CELL_ID.isin(match_tiles)]

    dw_path = create_directory_tree(root, f"data/{OUT_DIR}", "dynamic_world", str(YEAR))

    # %%
    params = [
        {
            'bbox':row[12].bounds, 
            'year': YEAR, 
            'path': dw_path, 
            'prefix': f'{row[1]}_', 
            'season':"leafon", 
            'overwrite': False
        } for row in qq_shp.itertuples()
    ]

    multithreaded_execution(get_dworld, params, WORKERS)
    
    # params[params == 'leafon'] = 'leafoff'
    # multithreaded_download(params, get_dworld)
