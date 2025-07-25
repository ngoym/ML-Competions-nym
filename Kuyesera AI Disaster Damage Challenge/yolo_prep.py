import os
import pandas as pd
import numpy as np
import yaml
import pdb
import json

# Geospatial libraries
from rasterio.mask import mask
from rasterio.plot import reshape_as_raster
from rasterio.transform import from_bounds
from shapely import wkt
from shapely.geometry import Polygon, box
from shapely.wkt import loads
import geopandas as gpd

from PIL import Image, ImageDraw



DATA_DIR = "/home/mutonkol/ML/KuyeserA/xview2/geotiffs"

x_train_dir = os.path.join(DATA_DIR, "train", "images")
y_train_dir = os.path.join(DATA_DIR, "train", "labels_json")

x_valid_dir = os.path.join(DATA_DIR, "test", "images")
y_valid_dir = os.path.join(DATA_DIR, "test", "labels_json")

TMP = f"{DATA_DIR}/train/tmp"
os.makedirs(TMP, exist_ok=True)

ids = [image_id for image_id in os.listdir(x_train_dir)  if image_id.endswith("_disaster.png")]
val_ids = [image_id for image_id in os.listdir(x_valid_dir)  if image_id.endswith("_disaster.png")]

def _clip_polygon_to_image(polygon, width, height):
        image_box = box(0, 0, width, height)
        return polygon.intersection(image_box)

def _polygon_to_mask(polygon, width, height):
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)

        if polygon.is_empty:
            return np.array(mask, dtype=np.uint8)

        if polygon.geom_type == "Polygon":
            if polygon.exterior is not None:
                x, y = polygon.exterior.coords.xy
                coords = [(xi, yi) for xi, yi in zip(x, y)]
                draw.polygon(coords, outline=1, fill=1)
        elif polygon.geom_type == "MultiPolygon":
            for poly in polygon.geoms:
                if poly.exterior is not None:
                    x, y = poly.exterior.coords.xy
                    coords = [(xi, yi) for xi, yi in zip(x, y)]
                    draw.polygon(coords, outline=1, fill=1)

        return np.array(mask, dtype=np.uint8)

def mask_to_bbox(mask):
    """
    Converts a binary mask to a bounding box.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W), where 1 indicates the object.

    Returns:
        list: Bounding box in [x_min, y_min, width, height] format.
    """
    # Find the coordinates of non-zero pixels
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Compute width and height
    width = x_max - x_min + 1
    height = y_max - y_min + 1

    return [x_min, y_min, x_max, y_max]

def create_labels_func(ids, yolo_label_dir, image_dir, labels_dir):
    labels = []
    count = 0
    _len = len(ids)
    damage_class_to_id = {
        "no-damage": 0,
        "minor-damage": 1,
        "major-damage": 2,
        "destroyed": 3,
    }

    for i in range(len(ids)):
        print(".",end="",flush=True)
        img = ids[i]
        #cat = cat_dict[df["class_s"].iloc[i]]
        label_path = f'{yolo_label_dir}/{img.replace(".png","")}.txt'
        image_path = f'{image_dir}/{img}'
        ilabel_path = f'{labels_dir}/{img.replace(".png","")}.json'

        img = Image.open(image_path)
        width, height = img.size
        # Get bounding boxes
        with open(ilabel_path, "r") as f:
            annotations = json.load(f)

        boxes = []
        labels = []
        masks = []
        annotations = annotations["features"]["xy"]
        if not annotations:
            print("Empty annotations", image_path)
            os.system(f"mv {image_path} {TMP}")
            continue
        for annotation in annotations:
            properties = annotation["properties"]
            subtype = properties.get("subtype", None)
            #if subtype is None: continue
            damage_label = damage_class_to_id.get(subtype, 0)
            #pdb.set_trace()

            polygon_wkt = annotation["wkt"]
            polygon = loads(polygon_wkt)

            if not polygon.is_valid:
                continue

            polygon = _clip_polygon_to_image(polygon, width, height)
            if polygon.is_empty:
                continue

            xmin, ymin, xmax, ymax = polygon.bounds
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(damage_label)

            mask = _polygon_to_mask(polygon, width, height)
            #bx = mask_to_bbox(mask)
            #print([xmin, ymin, xmax, ymax])
            #print(bx)
            #pdb.set_trace()
            masks.append(mask)
        #pdb.set_trace()
        if len(boxes) == 0: continue

        image = Image.open(image_path)
        imgx, imgy = image.size
        lbl = ''

        for i,bb in enumerate(boxes):
            #print(cats[i])
            cat = 0 #labels[i]

            #bb = eval(b)
            xcenter = ((bb[0] +  bb[2]) / 2) / imgx
            ycenter = ((bb[1] +  bb[3]) / 2) / imgy
            width = (bb[2] - bb[0]) / imgx
            height = (bb[3] - bb[1]) / imgy
            lbl += f'{cat} {xcenter:.4} {ycenter:.4} {width:.4} {height:.4}\n'
        if lbl:
            pass
            #with open(label_path, 'w') as F:
            #    F.write(lbl)
        else:
            a = None

        count += 1
        if count % 250 == 0:
            print(f"\n[INFO] - {count} of {_len} done.\n", flush=True)

print("\n\nCreating Train labels")
yolo_train_dir = os.path.join(DATA_DIR, "train", "labels")
os.makedirs(yolo_train_dir, exist_ok=True)
create_labels_func(ids, yolo_train_dir, x_train_dir, y_train_dir)
print("\n\nCreating Val labels")
yolo_valid_dir = os.path.join(DATA_DIR, "test", "labels")
os.makedirs(yolo_valid_dir, exist_ok=True)
create_labels_func(val_ids, yolo_valid_dir, x_valid_dir, y_valid_dir)

config = {
    #'names': ['no-damage', 'minor-damage', 'major-damage', 'destroyed'],
    'names': ['building'],
    'num_classes': 1,
    'train': x_train_dir,
    'val': x_valid_dir,
}

with open(f'data.yaml', 'w') as f:
    yaml.dump(config, f)