import os
import numpy as np
import pandas as pd
import rasterio
from rasterio import transform, warp


def load_sentinel(to_load):
    """Loads and transforms SENTINEL-2 image into a DataFrame.

    Parameters
    ----------
    to_load : str or arr
      path to raster or an in-memory array with SENTINEL-2 data

    Returns
    -------
    df : DataFrame
      flattened raster with additional derived fields
    """
    if isinstance(to_load, str):
        with rasterio.open(to_load) as src:
            s2 = src.read()
    elif isinstance(to_load, np.ndarray):
        s2 = to_load
    else:
        raise TypeError

    COLS = ['S2_B_LEAFOFF', 'S2_G_LEAFOFF', 'S2_R_LEAFOFF',
            'S2_RE1_LEAFOFF', 'S2_RE2_LEAFOFF', 'S2_RE3_LEAFOFF', 'S2_RE4_LEAFOFF',
            'S2_NIR_LEAFOFF', 'S2_SWIR1_LEAFOFF', 'S2_SWIR2_LEAFOFF',
            'S2_B_LEAFON', 'S2_G_LEAFON', 'S2_R_LEAFON',
            'S2_RE1_LEAFON', 'S2_RE2_LEAFON', 'S2_RE3_LEAFON', 'S2_RE4_LEAFON',
            'S2_NIR_LEAFON', 'S2_SWIR1_LEAFON', 'S2_SWIR2_LEAFON']
    df = pd.DataFrame(s2.reshape((20, -1)).T, columns=COLS, dtype='Int64')
    df = df.replace(0, np.nan) # nodata represented as zeros

    for season in ('LEAFOFF', 'LEAFON'):
        R, G, B = f'S2_R_{season}', f'S2_G_{season}', f'S2_B_{season}'
        NIR, SWIR1, SWIR2 =  f'S2_NIR_{season}', f'S2_SWIR1_{season}', f'S2_SWIR2_{season}'

        NDVI = f'S2_NDVI_{season}'
        df[NDVI] = (df[NIR] - df[R])/(df[NIR] + df[R])

        ENDVI = f'S2_ENDVI_{season}'
        df[NDVI] = (df[NIR] + df[G] - 2*df[B])/(df[NIR] + df[G] + 2*df[B])

        SAVI = f'S2_SAVI_{season}'
        df[SAVI] = 1.5*(df[NIR] - df[R])/(df[NIR] + df[R] + 0.5)

        BRIGHTNESS = f'S2_BRIGHTNESS_{season}'
        df[BRIGHTNESS] = 0.3029*df[B] + 0.2786*df[G] + 0.4733*df[R] + 0.5599*df[NIR] + 0.508*df[SWIR1] + 0.1872*df[SWIR2]

        GREENNESS = f'S2_GREENNESS_{season}'
        df[GREENNESS] = -0.2941*df[B] + -0.243*df[G] + -0.5424*df[R] + 0.7276*df[NIR] + 0.0713*df[SWIR1] + -0.1608*df[SWIR2]

        WETNESS = f'S2_WETNESS_{season}'
        df[WETNESS] = 0.1511*df[B] + 0.1973*df[G] + 0.3283*df[R] + 0.3407*df[NIR] + -0.7117*df[SWIR1] + -0.4559*df[SWIR2]

    df['S2_dB'] = df['S2_B_LEAFON'] - df['S2_B_LEAFOFF'].astype(int)
    df['S2_dG'] = df['S2_G_LEAFON'] - df['S2_G_LEAFOFF'].astype(int)
    df['S2_dR'] = df['S2_R_LEAFON'] - df['S2_R_LEAFOFF'].astype(int)
    df['S2_dRE1'] = df['S2_RE1_LEAFON'] - df['S2_RE1_LEAFOFF'].astype(int)
    df['S2_dRE2'] = df['S2_RE2_LEAFON'] - df['S2_RE2_LEAFOFF'].astype(int)
    df['S2_dRE3'] = df['S2_RE3_LEAFON'] - df['S2_RE3_LEAFOFF'].astype(int)
    df['S2_dRE4'] = df['S2_RE4_LEAFON'] - df['S2_RE4_LEAFOFF'].astype(int)
    df['S2_dNIR'] = df['S2_NIR_LEAFON'] - df['S2_NIR_LEAFOFF'].astype(int)
    df['S2_dSWIR1'] = df['S2_SWIR1_LEAFON'] - df['S2_SWIR1_LEAFOFF'].astype(int)
    df['S2_dSWIR2'] = df['S2_SWIR2_LEAFON'] - df['S2_SWIR2_LEAFOFF'].astype(int)
    df['S2_dNDVI'] = df['S2_NDVI_LEAFON'] - df['S2_NDVI_LEAFOFF'].astype(int)
    df['S2_dSAVI'] = df['S2_SAVI_LEAFON'] - df['S2_SAVI_LEAFOFF'].astype(int)
    df['S2_dBRIGHTNESS'] = df['S2_BRIGHTNESS_LEAFON'] - df['S2_BRIGHTNESS_LEAFOFF']
    df['S2_dGREENNESS'] = df['S2_GREENNESS_LEAFON'] - df['S2_GREENNESS_LEAFOFF']
    df['S2_dWETNESS'] = df['S2_WETNESS_LEAFON'] - df['S2_WETNESS_LEAFOFF']

    return df

def load_landtrendr(to_load):
    """Loads and transforms a Landtrendr-derived raster into
    into a DataFrame.

    Parameters
    ----------
    to_load : str or arr
      path to raster or an in-memory array with DEM data

    Returns
    -------
    df : DataFrame
      flattened raster with additional derived fields
    """
    if isinstance(to_load, str):
        with rasterio.open(to_load) as src:
            lt = src.read()
    elif isinstance(to_load, np.ndarray):
        lt = to_load
    else:
        raise TypeError

    COLS = [
        'LT_YSD_SWIR1', 'LT_MAG_SWIR1', 'LT_DUR_SWIR1', 'LT_RATE_SWIR1',
        'LT_YSD_NBR', 'LT_MAG_NBR', 'LT_DUR_NBR', 'LT_RATE_NBR'
    ]

    df = (pd.DataFrame(lt.reshape([8,-1]).T,
                      columns=COLS, dtype='Int64')
          .replace(-32768, np.nan))  # nodata represented as -32768

    return df

def load_dem(to_load, meta=None):
    """Loads and transforms a Digital Elevation Model (DEM) image
    into a DataFrame.

    Parameters
    ----------
    to_load : str or arr
      path to raster or an in-memory array with DEM data
    meta : dict, optional
      dictionary of raster attributes, must include width,
      height, transform, and crs

    Returns
    -------
    df : DataFrame
      flattened raster with additional derived fields
    """
    if isinstance(to_load, str):
        with rasterio.open(to_load) as src:
            dem = src.read()
            meta = src.meta
    elif isinstance(to_load, np.ndarray) and meta is not None:
        dem = to_load

    else:
        raise TypeError

    df = pd.DataFrame(columns=['elevation', 'lat', 'lon'])

    df['elevation'] = dem.ravel()
    df['elevation'] = df['elevation'].astype('Int64')

    # fetch lat and lon for each pixel in a raster
    rows, cols = np.indices((meta['height'], meta['width']))
    xs, ys = transform.xy(meta['transform'], cols.ravel(), rows.ravel())
    lons, lats = warp.transform(meta['crs'], {'init':'EPSG:4326'}, xs, ys)
    df['lat'] = lats
    df['lon'] = lons

    # nodata represented as -32768
    df.loc[df.elevation == -32768] = np.nan

    return df


def load_features(path_to_rasters, cell_id):
    """Loads data from disk into a dataframe ready for predictive modeling.

    Parameters
    ----------
    path_to_rasters : str
      path to the directory where subdirectories for 'sentinel', 'landtrendr',
      and 'dem' imagery can be found.
    cell_id : int or str
      cell id which identifies a quarter quad.

    Returns
    -------
    df : DataFrame
      dataframe with feature data ready for predictive modeling
    """
    s2_path = os.path.join(path_to_rasters, 'sentinel', f'{cell_id}_sentinel.tif')
    lt_path = os.path.join(path_to_rasters, 'landtrendr', f'{cell_id}_landtrendr.tif')
    dem_path = os.path.join(path_to_rasters, 'dem', f'{cell_id}_dem.tif')

    s2 = load_sentinel(s2_path)
    lt = load_landtrendr(lt_path)
    dem = load_dem(dem_path)

    df = pd.concat([s2, lt, dem], axis=1)

    return df
