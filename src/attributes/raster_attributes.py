
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio import transform, warp

def s2_atts(s2):
    if type(s2) == str:
        with rasterio.open(s2) as src:
            s2 = src.read()
    elif type(s2) == np.ndarry:
        s2 = s2
        
    df = pd.DataFrame(s2.reshape([12,-1]).T)
    df.columns = ['S2_B_LEAFOFF', 'S2_G_LEAFOFF', 'S2_R_LEAFOFF', 'S2_NIR_LEAFOFF', 
                  'S2_SWIR1_LEAFOFF', 'S2_SWIR2_LEAFOFF',
                  'S2_B_LEAFON', 'S2_G_LEAFON', 'S2_R_LEAFON', 'S2_NIR_LEAFON', 
                  'S2_SWIR1_LEAFON', 'S2_SWIR2_LEAFON']    
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
  
    df['S2_dR'] = df['S2_R_LEAFON'] - df['S2_R_LEAFOFF'].astype(int)
    df['S2_dG'] = df['S2_G_LEAFON'] - df['S2_G_LEAFOFF'].astype(int)
    df['S2_dB'] = df['S2_B_LEAFON'] - df['S2_B_LEAFOFF'].astype(int)
    df['S2_dNIR'] = df['S2_NIR_LEAFON'] - df['S2_NIR_LEAFOFF'].astype(int)
    df['S2_dSWIR1'] = df['S2_SWIR1_LEAFON'] - df['S2_SWIR1_LEAFOFF'].astype(int)
    df['S2_dSWIR2'] = df['S2_SWIR2_LEAFON'] - df['S2_SWIR2_LEAFOFF'].astype(int)
    df['S2_dNDVI'] = df['S2_NDVI_LEAFON'] - df['S2_NDVI_LEAFOFF'].astype(int) 
    df['S2_dSAVI'] = df['S2_SAVI_LEAFON'] - df['S2_SAVI_LEAFOFF'].astype(int)
    df['S2_dBRIGHTNESS'] = df['S2_BRIGHTNESS_LEAFON'] - df['S2_BRIGHTNESS_LEAFOFF']
    df['S2_dGREENNESS'] = df['S2_GREENNESS_LEAFON'] - df['S2_GREENNESS_LEAFOFF']
    df['S2_dWETNESS'] = df['S2_WETNESS_LEAFON'] - df['S2_WETNESS_LEAFOFF']

    return df

def lt_atts(lt):
    if type(lt) == str:
        with rasterio.open(lt) as src:
            lt = src.read()
    elif type(s2) == np.ndarry:
        lt = lt
    df = pd.DataFrame(lt.reshape([8,-1]).T)
    df.columns = ['LT_YSD_SWIR1', 'LT_MAG_SWIR1', 'LT_DUR_SWIR1', 'LT_RATE_SWIR1',
                'LT_YSD_NBR', 'LT_MAG_NBR', 'LT_DUR_NBR', 'LT_RATE_NBR']
    return df


def dem_atts(dem):
    if type(dem) == str: 
        with rasterio.open(dem) as src:
            dem = src.read()
            meta = src.meta
    elif type(dem) == np.ndarry:
        dem = dem
        meta = dem.meta

    array = np.array(dem)
    df = pd.DataFrame()

    df['elevation'] = array.ravel() 

    # fetch lat and lon for each pixel in a raster
    rows, cols = np.indices((meta['height'], meta['width']))
    xs, ys = transform.xy(meta['transform'], cols.ravel(), rows.ravel())
    lons, lats = warp.transform(meta['crs'], {'init':'EPSG:4326'}, xs, ys)
    df['lat'] = lats
    df['lon'] = lons
    
    return df


def format_as_df(s2, lt, dem):
 
    s2_df = s2_atts(s2)
    lt_df = lt_atts(lt)
    dem_df = dem_atts(dem)
    
    dfs = [s2_df, lt_df, dem_df]
    
    df = pd.concat(dfs, axis=1)
    
    return df
        
