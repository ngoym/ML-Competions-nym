required_bands = {
        "B3": "B03.tif",  # Green
        "B4": "B04.tif",  # Red
        "B8": "B08.tif",  # NIR (10m)
        "B11": "B11.tif",  # SWIR1 (20m)
        "B12": "B12.tif",  # SWIR2 (20m)
    }

import os
import rasterio
import numpy as np
import pandas as pd
import pdb

pre = "pre-tifs"
post = "post-tifs"

def read_band(file_path, band_number):
    with rasterio.open(file_path) as src:
        band_data = src.read(band_number)
    return band_data

bands = ["B3", "B4", "B8"]
eps = 1e-10
# Pre NVDI
pre_nir = read_band(os.path.join(pre, required_bands["B8"]), 1)
pre_red = read_band(os.path.join(pre, required_bands["B4"]), 1)
pre_green = read_band(os.path.join(pre, required_bands["B3"]), 1)
pre_ndvi = np.mean((pre_nir - pre_red) / (pre_nir + pre_red + eps))
pre_ndwi = np.mean((pre_green - pre_nir) / (pre_green + pre_nir + eps))
# Post NVDI
post_nir = read_band(os.path.join(post, required_bands["B8"]), 1)
post_red = read_band(os.path.join(post, required_bands["B4"]), 1)
post_green = read_band(os.path.join(post, required_bands["B3"]), 1)
post_ndvi = np.mean((post_nir - post_red) / (post_nir + post_red + eps))
post_ndwi = np.mean((post_green - post_nir) / (post_green + post_nir + eps))


print(pre_ndvi, post_ndvi)
print(pre_ndwi, post_ndwi)


df = pd.read_csv("train.csv")
s2df = pd.read_csv("train_s2.csv")
test_df = pd.read_csv("test_df.csv")
sd = s2df.dropna().reset_index(drop=True)
df["NDVI_mean"] = np.nan
test_df["NDVI_mean"] = post_ndvi

#for i in range(len(sd)):
#    for j in range(len(df)):
#        if sd.at[i,'id'] in df.at[j,'pre_image']:
#            df.at[j,'NDVI_mean'] = sd.at[i,'NDVI_mean']

df['pre_image_id'] = df['pre_image'].str.extract(r'(.*)_pre_disaster')[0]

df = df.merge(sd[['id', 'NDVI_mean']], how='left', left_on='pre_image_id', right_on='id')
df['NDVI_mean'] = df['NDVI_mean_y'].fillna(df['NDVI_mean_x'])
df['NDVI_mean'] = df['NDVI_mean'].fillna(0.4)
df.drop(columns=['NDVI_mean_x', 'NDVI_mean_y', 'id', 'pre_image_id'], inplace=True)
df.to_csv("train_ndvi.csv", index=False)
test_df.to_csv("test_ndvi.csv", index=False)