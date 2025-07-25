import rasterio
from PIL import Image
import numpy as np
from pathlib import Path
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, box
from rasterio.mask import mask
import pdb

TEST = True

# Extract and add lat and lon bounds - to be used in downloading data from aws
def parse_coords(coord_str):
    # Remove parentheses
    coord_str = coord_str.strip("()")
    # Split by the semicolon
    lat_str, lon_str = coord_str.split(";")
    # Convert to float
    lat = float(lat_str)
    lon = float(lon_str)
    return lat, lon

def clip_image(image_path, corners):
    """
    Clip a GeoTIFF using four corner coordinates

    Args:
        image_path (str): Path to the GeoTIFF file
        corners (list): List of (lat,lon) coordinates defining the polygon corners
    """
    # Swap lat/lon to lon/lat for each corner
    swapped_corners = [(lon, lat) for lat, lon in corners]

    with rasterio.open(image_path) as src:
        # Create a polygon from the swapped corners
        polygon = Polygon(swapped_corners)

        # Create a GeoDataFrame with the polygon
        geo = gpd.GeoDataFrame({"geometry": [polygon]}, crs="EPSG:4326")

        # Perform the clip
        out_image, out_transform = mask(src, geo.geometry, crop=False)

        # Copy the metadata
        out_meta = src.meta.copy()

        # Update metadata
        out_meta.update(
            {
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )

        return out_image, out_meta

if TEST:
    DIR = '/home/mutonkol/ML/KuyeserA/tif_test/Images'
    test_coords = pd.read_csv("test_image_coords.csv")
    test_pre_df = pd.Series(
        [
            x
            for x in Path(DIR).glob("*.tif")
            if x.parts[-1].split(".")[0].split("_")[-2] == "pre"
        ]
    ).to_frame(name="pre_image_path")
    test_post_df = pd.Series(
        [
            x
            for x in Path(DIR).glob("*.tif")
            if x.parts[-1].split(".")[0].split("_")[-2] == "post"
        ]
    ).to_frame(name="post_image_path")

    test_pre_df["id"] = [
        "_".join(x.parts[-1].split(".")[0].split("_")[:-2])
        for x in test_pre_df.pre_image_path
    ]
    test_post_df["id"] = [
        "_".join(x.parts[-1].split(".")[0].split("_")[:-2])
        for x in test_post_df.post_image_path
    ]

    test_df = test_pre_df.merge(test_post_df, how="left", on="id")
    test_df = test_coords.merge(test_df, how="left", on="id")
    test_df["flood_name"] = ["_".join(x.split("_")[:-1]) for x in test_df.id]

    # Add columns for min/max lat/lon
    test_df["post_min_lon"] = None
    test_df["post_max_lon"] = None
    test_df["post_min_lat"] = None
    test_df["post_max_lat"] = None

    for idx, row in test_df.iterrows():
        # Parse all four corner coordinates
        corners = [
            parse_coords(row["pre_top_left"]),
            parse_coords(row["pre_top_right"]),
            parse_coords(row["pre_bottom_right"]),
            parse_coords(row["pre_bottom_left"]),
        ]

        # Extract all latitudes and longitudes into separate lists
        lats = [c[0] for c in corners]
        lons = [c[1] for c in corners]

        # Compute min/max latitude and longitude
        min_lat = min(lats)
        max_lat = max(lats)
        min_lon = min(lons)
        max_lon = max(lons)

        # Assign to DataFrame
        test_df.at[idx, "post_min_lat"] = min_lat
        test_df.at[idx, "post_max_lat"] = max_lat
        test_df.at[idx, "post_min_lon"] = min_lon
        test_df.at[idx, "post_max_lon"] = max_lon

    for i in range(len(test_df)):
        print(".", end="", flush=True)
        row = test_df.loc[i]
        pre_image_path = row.pre_image_path
        post_image_path = row.post_image_path
        id = row.id
        #pdb.set_trace()
        pre_basename = pre_image_path.parts[-1]
        post_basename = post_image_path.parts[-1]

        # Get corner coordinates
        pre_corners = [
            eval(row.pre_top_left.replace(";", ",")),
            eval(row.pre_top_right.replace(";", ",")),
            eval(row.pre_bottom_right.replace(";", ",")),
            eval(row.pre_bottom_left.replace(";", ",")),
        ]

        post_corners = [
            eval(row.post_top_left.replace(";", ",")),
            eval(row.post_top_right.replace(";", ",")),
            eval(row.post_bottom_right.replace(";", ",")),
            eval(row.post_bottom_left.replace(";", ",")),
        ]

        # Clip images
        pre_clipped, pre_meta = clip_image(pre_image_path, pre_corners)
        post_clipped, post_meta = clip_image(post_image_path, post_corners)

        pre_clipped = pre_clipped.transpose(1, 2, 0)
        post_clipped = post_clipped.transpose(1, 2, 0)

        pre_clipped = Image.fromarray(pre_clipped)
        post_clipped = Image.fromarray(post_clipped)

        pre_clipped.save(f"{DIR}/{pre_basename.replace('.tif', '.png')}")
        post_clipped.save(f"{DIR}/{post_basename.replace('.tif', '.png')}")
        os.remove(pre_image_path)
        os.remove(post_image_path)

else:
    DIR = '/home/mutonkol/ML/KuyeserA/xview2/geotiffs/hold/images'
    tifs = [t for t  in os.listdir(DIR) if t.endswith('.tif')]

    os.chdir(DIR)
    for tif in tifs:
        print(".", end="", flush=True)
        try:
            with rasterio.open(tif) as src:
                data = src.read()
                if src.count >= 3:
                # Take the first three bands for RGB
                    rgb = np.stack([data[0], data[1], data[2]], axis=-1)
                else:
                    # If single band, just replicate to RGB (grayscale)
                    rgb = np.stack([data[0]] * 3, axis=-1)

                # Normalize data to 0-255 (uint8)
                rgb_normalized = (255 * (rgb - rgb.min()) / (rgb.max() - rgb.min())).astype(np.uint8)
                img = Image.fromarray(rgb_normalized)
                img.save(tif.replace('.tif', '.png'))
                os.remove(tif)
        except Exception as e:
            print(f"Error processing {tif}: {e}")
