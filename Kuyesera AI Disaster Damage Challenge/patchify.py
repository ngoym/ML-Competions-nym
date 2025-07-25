import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import pdb

def create_patches(img_file, label_file, pre_dir, post_dir):
    img_pre = cv2.imread(img_file)
    img_post = cv2.imread(img_file.replace("_pre_disaster.png", "_post_disaster.png"))
    image = Image.open(img_file)
    imgx, imgy = image.size
    lbls = []
    boxes = []
    pre_img_id = []
    post_img_id = []
    img_name = os.path.basename(img_file).replace("_pre_disaster.png", "")
    patch_count = 0
    with open(label_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.split()
            x1 = (float(line[1]) - float(line[3])/2) * imgx
            y1 = (float(line[2]) - float(line[4])/2) * imgy
            x2 = (float(line[1]) + float(line[3])/2) * imgx
            y2 = (float(line[2]) + float(line[4])/2) * imgy
            # Remove very tiny boxes
            #print("[DEBUG] - x1,x2,y1,y2,imgx, imgy, patch_count", x1, x2, y1, y2, imgx, imgy, patch_count)
            #if int(x1) == int(x2) or int(y1) == int(y2):
            #    continue
            #if patch_count == 5:
            #    pdb.set_trace()
            try:
                crop_pre = img_pre[int(y1):int(y2), int(x1):int(x2)]
                crop_post = img_post[int(y1):int(y2), int(x1):int(x2)]
                cv2.imwrite(os.path.join(pre_dir, img_name + f"_pre_disaster_{patch_count}.png"), crop_pre)
                cv2.imwrite(os.path.join(post_dir, img_name + f"_post_disaster_{patch_count}.png"), crop_post)
                lbls.append(int(line[0]))
                pre_img_id.append(img_name + f"_pre_disaster_{patch_count}.png")
                post_img_id.append(img_name + f"_post_disaster_{patch_count}.png")
            except Exception as e:
                print(f"[ERROR]: {e}", flush=True)
                print("[DEBUG] - img_file, label_file", img_file, label_file, flush=True)
                print("[DEBUG] - x1,x2,y1,y2,imgx, imgy, patch_count", x1, x2, y1, y2, imgx, imgy, patch_count, flush=True)

            patch_count += 1
    return lbls, pre_img_id, post_img_id

DATA_DIR = "/home/mutonkol/ML/KuyeserA/xview2/geotiffs"
x_train_dir = os.path.join(DATA_DIR, "train", "images")
y_train_dir = os.path.join(DATA_DIR, "train", "labels")

pre_disaster = [image_id.replace("_pre_disaster.png","") for image_id in os.listdir(x_train_dir)  if image_id.endswith("_pre_disaster.png")]
labels = [image_id.replace("_post_disaster.txt","") for image_id in os.listdir(y_train_dir) if image_id.endswith("_post_disaster.txt")]

img_ids = sorted(list(set(pre_disaster).intersection(set(labels))))
pre_dir = os.path.join(DATA_DIR, "train", "pre_disaster")
post_dir = os.path.join(DATA_DIR, "train", "post_disaster")
os.makedirs(pre_dir, exist_ok=True)
os.makedirs(post_dir, exist_ok=True)

train_labels = []
train_pre_images = []
train_post_images = []
c = 0
max_count = len(img_ids)
for img_id in img_ids:
    c += 1
    print(".",end="",flush=True)
    img_file = os.path.join(x_train_dir, img_id + "_pre_disaster.png")
    label_file = os.path.join(y_train_dir, img_id + "_post_disaster.txt")
    lbls, pre_img_id, post_img_id = create_patches(img_file, label_file, pre_dir, post_dir)
    train_labels.extend(lbls)
    train_pre_images.extend(pre_img_id)
    train_post_images.extend(post_img_id)

    if c % 250 == 0:
        print(f"\nCreated {c}/{max_count} sets", flush=True)

df = pd.DataFrame({
        "pre_image": train_pre_images,
        "post_image": train_post_images,
        "label": train_labels,
})

df.to_csv("train.csv", index=False)