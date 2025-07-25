import pdb
# Geospatial libraries
import geopandas as gpd
import pyproj
import pyproj.datadir
import pystac_client
import rasterio
import rioxarray
import xarray as xr
from pyproj import CRS, Transformer
from rasterio.mask import mask
from rasterio.plot import reshape_as_raster
from rasterio.transform import from_bounds
from shapely import wkt
from shapely.geometry import Polygon, box
from shapely.wkt import loads
import warnings
warnings.filterwarnings("ignore")
# AWS
import boto3
from botocore import UNSIGNED
from botocore.client import Config

# Standard library imports
import json
import multiprocessing
import os
import warnings
from collections import Counter
from functools import partial
from pathlib import Path


# Data manipulation and analysis
import numpy as np
import pandas as pd
from tqdm import tqdm

# Visualization
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import pdb

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Create a df with all the image and label paths
DIR = "/home/mutonkol/ML/KuyeserA/xview2/geotiffs"

# Load test image coords and image paths
test_coords = pd.read_csv( "test_image_coords.csv")
test_pre_df = pd.Series(
    [
        x
        for x in Path("Images").glob("*.tif")
        if x.parts[-1].split(".")[0].split("_")[-2] == "pre"
    ]
).to_frame(name="pre_image_path")
test_post_df = pd.Series(
    [
        x
        for x in Path("Images").glob("*.tif")
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


# Preview test df
print(test_df.head())

dfs = []
for dataset_path in tqdm(os.listdir(DIR)):
    image_paths = (Path(DIR) / dataset_path / "images").glob("*.png")
    images_df = pd.Series(image_paths).to_frame(name="image_path")
    images_df["id"] = [
        "_".join(x.parts[-1].split(".")[0].split("_")[:-2])
        for x in images_df.image_path
    ]
    images_df["pre_post"] = [
        x.parts[-1].split(".")[0].split("_")[-2] for x in images_df.image_path
    ]

    label_paths = (Path(DIR) / dataset_path / "labels_json").glob("*.json")
    labels_df = pd.Series(label_paths).to_frame(name="label_path")
    labels_df["id"] = [
        "_".join(x.parts[-1].split(".")[0].split("_")[:-2])
        for x in labels_df.label_path
    ]
    labels_df["pre_post"] = [
        x.parts[-1].split(".")[0].split("_")[-2] for x in labels_df.label_path
    ]

    merged_df = images_df.merge(labels_df, how="left", on=["id", "pre_post"])
    merged_df = merged_df[["id", "pre_post", "image_path", "label_path"]]
    merged_df["dataset_type"] = dataset_path

    pre_df = merged_df[merged_df.pre_post == "pre"]
    post_df = merged_df[merged_df.pre_post == "post"]

    pre_df.columns = [f"pre_{x.replace('-', '_')}" for x in pre_df.columns]
    pre_df = pre_df.drop(["pre_pre_post"], axis=1)
    pre_df = pre_df.rename(columns={"pre_id": "id", "pre_dataset_type": "dataset_type"})

    post_df.columns = [f"post_{x.replace('-', '_')}" for x in post_df.columns]
    post_df = post_df.drop(["post_pre_post"], axis=1)
    post_df = post_df.rename(
        columns={"post_id": "id", "post_dataset_type": "dataset_type"}
    )

    merged_df = pre_df.merge(post_df, how="left", on=["id", "dataset_type"])

    dfs.append(merged_df)

df = pd.concat(dfs, ignore_index=True)
df["flood_name"] = ["_".join(x.split("_")[:-1]) for x in df.id]

def get_overall_bounds(label_path):

    global_minx = float("inf")
    global_miny = float("inf")
    global_maxx = float("-inf")
    global_maxy = float("-inf")
    try:
        with open(label_path, "r") as file:
            data = json.load(file)

        for poly_data in data["features"]["lng_lat"]:
            polygon = wkt.loads(poly_data["wkt"])
            minx, miny, maxx, maxy = polygon.bounds

            # Update global bounds
            if minx < global_minx:
                global_minx = minx
            if miny < global_miny:
                global_miny = miny
            if maxx > global_maxx:
                global_maxx = maxx
            if maxy > global_maxy:
                global_maxy = maxy
    except:
        print("[ERROR]", label_path)
    # minx = min_lon, miny = min_lat, maxx = max_lon, maxy = max_lat
    return global_minx, global_miny, global_maxx, global_maxy

df[["post_min_lon", "post_min_lat", "post_max_lon", "post_max_lat"]] = df[
    "post_label_path"
].apply(
    lambda path: pd.Series(
        get_overall_bounds(path),
        index=["post_min_lon", "post_min_lat", "post_max_lon", "post_max_lat"],
    )
)
print(df.shape)

# Create targets
# Function to count subtypes from a label path
def get_subtype_counts(label_path):
    with open(label_path, "r") as file:
        label_data = json.load(file)

    subtypes = [
        building["properties"]["subtype"] for building in label_data["features"]["xy"]
    ]
    return Counter(subtypes)


# Process each label_path and aggregate counts
all_counts = []
for label_path in df["post_label_path"]:
    subtype_counts = get_subtype_counts(label_path)
    all_counts.append(subtype_counts)

# Convert counts to DataFrame and combine with original DataFrame
counts_df = (
    pd.DataFrame(all_counts).fillna(0).astype(int)
)  # Fill NaN with 0 and cast to int
df = pd.concat([df, counts_df], axis=1)

# Drop un-classified, as the main target categories include no-damage, minor-damage, major-damage, destroyed
df = df.drop("un-classified", axis=1)
print(df.head())

# Add disaster dates - Search dates over the internet
disaster_dates = {
    "hurricane-michael": {"start_date": "2018-10-07", "end_date": "2018-10-16"},
    "guatemala-volcano": {"start_date": "2018-06-03", "end_date": "2018-07-06"},
    "midwest-flooding": {"start_date": "2019-03-15", "end_date": "2019-04-01"},
    "hurricane-matthew": {"start_date": "2016-09-28", "end_date": "2016-10-10"},
    "hurricane-harvey": {"start_date": "2017-08-17", "end_date": "2017-09-02"},
    "hurricane-florence": {"start_date": "2018-08-31", "end_date": "2018-09-18"},
    "palu-tsunami": {"start_date": "2018-09-28", "end_date": "2018-10-28"},
    "socal-fire": {"start_date": "2018-11-08", "end_date": "2018-11-25"},
    "mexico-earthquake": {"start_date": "2017-09-19", "end_date": "2017-09-19"},
    "santa-rosa-wildfire": {"start_date": "2017-10-08", "end_date": "2017-10-31"},
    "malawi-cyclone": {"start_date": "2023-03-11", "end_date": "2023-03-19"},
}

# Extract start and end dates into separate dictionaries
start_dates = {k: v["start_date"] for k, v in disaster_dates.items()}
end_dates = {k: v["end_date"] for k, v in disaster_dates.items()}

df["start_date"] = df["flood_name"].map(start_dates)
df["end_date"] = df["flood_name"].map(end_dates)
df["end_date"] = df.end_date.fillna('2018-08-03')
df["start_date"] = df.end_date.fillna('2018-06-03')

# Class to download S2 NDVI, NBR, NDWI, and NDMI indices from aws
# Feel free to use other bands to create more features
# Parallelize for faster downloads
class S3SatelliteDataFetcher:
    def __init__(
        self,
        api_url="https://earth-search.aws.element84.com/v1",
        bucket_name="sentinel-cogs",
    ):
        """Initialize STAC client and S3 client"""
        self.client = pystac_client.Client.open(api_url)
        self.s3_client = boto3.client(
            "s3", config=Config(signature_version=UNSIGNED), region_name="us-west-2"
        )
        self.bucket_name = bucket_name

        # Initialize the pyproj database
        pyproj.datadir.set_data_dir("/usr/share/proj")

    def get_https_url(self, s3_path):
        """Convert S3 path to HTTPS URL"""
        path = s3_path.replace("s3://sentinel-cogs/", "")
        return f"https://sentinel-cogs.s3.us-west-2.amazonaws.com/{path}"

    def search_scenes(
        self,
        geometry,
        date_range,
        cloud_cover=20,
        collection="sentinel-2-l2a",
        max_items=1,
    ):
        """Search for satellite scenes using STAC API"""
        query = {"eo:cloud_cover": {"lt": cloud_cover}}
        datetime_str = f"{date_range[0]}/{date_range[1]}"

        search = self.client.search(
            collections=[collection],
            intersects=geometry,
            datetime=datetime_str,
            query=query,
            max_items=max_items,
        )

        items = list(search.get_items())  # Convert iterator to list
        s3_paths = []
        for item in items:
            if "earthsearch:s3_path" in item.properties:
                s3_paths.append(item.properties["earthsearch:s3_path"])

        return items, s3_paths

    def download_raster(self, url_or_s3_path, bbox=None):
        """Download raster data from URL or S3 path with proper coordinate handling."""
        if url_or_s3_path.startswith("s3://"):
            url = self.get_https_url(url_or_s3_path)
        else:
            url = url_or_s3_path

        try:
            raster = rioxarray.open_rasterio(url)
        except:
            return None

        # Check CRS
        dst_crs = raster.rio.crs
        if dst_crs is None:
            # Assume a fallback CRS
            dst_crs = CRS.from_epsg(32632)
            raster = raster.rio.write_crs(dst_crs)

        if bbox is not None:
            try:
                # Source CRS: WGS84
                src_crs = CRS.from_epsg(4326)

                # Transform coordinates to the raster CRS
                transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
                minx, miny = transformer.transform(bbox[0], bbox[1])
                maxx, maxy = transformer.transform(bbox[2], bbox[3])

                # Clip the raster
                clipped_raster = raster.rio.clip_box(
                    minx=minx, miny=miny, maxx=maxx, maxy=maxy
                )
                # Check if the clipped data is empty
                if np.all(np.isnan(clipped_raster.values)):
                    return None
                raster = clipped_raster
            except:
                # Return full raster if clipping fails
                pass

        return raster


def get_scene_bands(
    min_lon, min_lat, max_lon, max_lat, start_date, end_date, cloud_cover=20
):
    fetcher = S3SatelliteDataFetcher()
    bbox = (min_lon, min_lat, max_lon, max_lat)
    polygon = box(min_lon, min_lat, max_lon, max_lat)

    items, s3_paths = fetcher.search_scenes(
        geometry=polygon,
        date_range=(start_date, end_date),
        cloud_cover=cloud_cover,
        max_items=1,
    )

    if not s3_paths:
        return None

    s3_path = s3_paths[0]

    required_bands = {
        "B3": "B03.tif",  # Green
        "B4": "B04.tif",  # Red
        "B8": "B08.tif",  # NIR (10m)
        "B11": "B11.tif",  # SWIR1 (20m)
        "B12": "B12.tif",  # SWIR2 (20m)
    }

    band_data = {}
    for b, filename in required_bands.items():
        url = f"{s3_path}/{filename}"
        raster = fetcher.download_raster(url, bbox=bbox)
        if raster is None:
            return None
        raster = raster.squeeze()
        if np.all(np.isnan(raster.values)):
            return None
        band_data[b] = raster

    # Use B8 as the reference band since it's 10m resolution
    ref_band = band_data["B8"]

    # Reproject/align each band to match B8
    for b in band_data:
        if b == "B8":
            continue
        band_data[b] = band_data[b].rio.reproject_match(ref_band)
        if np.all(np.isnan(band_data[b].values)):
            return None

    return band_data


def compute_indices(band_data):
    """
    Compute NDVI, NBR, NDWI, and NDMI from the band data.
    Each band_data[b] is a DataArray.
    Return a dict with mean values of these indices.
    """
    B3 = band_data["B3"].values.astype(np.float32)
    B4 = band_data["B4"].values.astype(np.float32)
    B8 = band_data["B8"].values.astype(np.float32)
    B11 = band_data["B11"].values.astype(np.float32)
    B12 = band_data["B12"].values.astype(np.float32)

    # Avoid division by zero
    eps = 1e-10

    # Compute indices
    NDVI = (B8 - B4) / (B8 + B4 + eps)
    NBR = (B8 - B12) / (B8 + B12 + eps)
    NDWI = (B3 - B8) / (B3 + B8 + eps)
    NDMI = (B8 - B11) / (B8 + B11 + eps)

    # Compute mean values
    return {
        "NDVI_mean": float(np.nanmean(NDVI)),
        "NBR_mean": float(np.nanmean(NBR)),
        "NDWI_mean": float(np.nanmean(NDWI)),
        "NDMI_mean": float(np.nanmean(NDMI)),
    }


def process_row(row, cloud_cover=20):
    """
    Process a single row from the DataFrame.
    Retrieves band data, computes indices, and returns a dictionary of values.
    If no scene is found or bands are invalid, return None values.
    """
    img_id = row["id"]
    min_lon = row["post_min_lon"]
    min_lat = row["post_min_lat"]
    max_lon = row["post_max_lon"]
    max_lat = row["post_max_lat"]
    start_date = str(row["start_date"])
    end_date = str(row["end_date"])

    band_data = get_scene_bands(
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        start_date,
        end_date,
        cloud_cover=cloud_cover,
    )
    if band_data is None:
        return {
            "id": img_id,
            "NDVI_mean": None,
            "NBR_mean": None,
            "NDWI_mean": None,
            "NDMI_mean": None,
        }

    indices = compute_indices(band_data)
    indices["id"] = img_id
    return indices


def parallel_process(df, cloud_cover=20, n_processes=4):
    """
    Parallelize processing of rows to compute NDVI, NBR, NDWI, NDMI for each id.
    Returns a DataFrame with these values.
    """
    rows = [row for _, row in df.iterrows()]
    func = partial(process_row, cloud_cover=cloud_cover)

    results = []
    with multiprocessing.Pool(processes=n_processes) as pool:
        for res in tqdm(
            pool.imap_unordered(func, rows), total=len(rows), desc="Processing"
        ):
            results.append(res)

    results_df = pd.DataFrame(results)
    # If 'id' exists in both dataframes, align the result.
    if "id" in df.columns and "id" in results_df.columns:
        results_df = results_df.set_index("id").reindex(df["id"]).reset_index()

    return results_df

train_s2_df = parallel_process(df, n_processes=8)
train_s2_df.to_csv("train_s2.csv", index=False)

# Process test data
# For test data, Malawi landcover info can be found from the landsat dataset
# The resolution is large and given the concentration of the flood area, it is sufficient
# to get only a single datapoint from here.
# We opt to download the tif data from the landsat dataset by hand as it is only a handful of images
# and the data is not too large.
# The downloaded data is provided in the pre-tifs and post-tifs folders.
# The remainder of the script will just print the urls to download the data from.

test_df["start_date"] = test_df["flood_name"].map(start_dates)
test_df["end_date"] = test_df["flood_name"].map(end_dates)

test_df.to_csv("test_df.csv", index=False)

import satsearch
from pystac_client import Client
LandsatSTAC = Client.open("https://landsatlook.usgs.gov/stac-server", headers=[])
SentinelSTAC = satsearch.Search.search( url = "https://earth-search.aws.element84.com/v0" )
print("Found " + str(SentinelSTAC.found()) + "items")

def BuildSquare(lon, lat, delta):
    c1 = [lon + delta, lat + delta]
    c2 = [lon + delta, lat - delta]
    c3 = [lon - delta, lat - delta]
    c4 = [lon - delta, lat + delta]
    geometry = {"type": "Polygon", "coordinates": [[ c1, c2, c3, c4, c1 ]]}
    return geometry

geometry = BuildSquare(test_df.post_min_lon[0], test_df.post_min_lat[0], 0.01)
timeRange = '2023-03-11/2023-03-14'
LandsatSearch = LandsatSTAC.search (
    intersects = geometry,
    datetime = timeRange,
    query =  ['eo:cloud_cover95'],
    collections = ["landsat-c2l2-sr"] )

Landsat_items = [i.to_dict() for i in LandsatSearch.get_items()]
print(f"{len(Landsat_items)} Landsat scenes fetched")

for item in Landsat_items:
    red_href = item['assets']['red']['href']
    red_s3 = item['assets']['red']['alternate']['s3']['href']
    print(red_href)
    print(red_s3)
# This will list items to download.
# Easy to just paste the URL into a browser to download the data.
SentinelSearch = satsearch.Search.search(
    url = "https://earth-search.aws.element84.com/v0",
    intersects = geometry,
    datetime = timeRange,
    collections = ['sentinel-s2-l2a-cogs'] )

Sentinel_items = SentinelSearch.items()
print(Sentinel_items.summary())

for item in Sentinel_items:
    red_s3 = item.assets['B04']['href']
    print(red_s3)

item = Sentinel_items[0]
print(item.assets.keys())


