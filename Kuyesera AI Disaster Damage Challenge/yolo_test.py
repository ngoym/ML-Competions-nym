from ultralytics import YOLO, RTDETR
import os
import pdb
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import pickle
import sys
from ensemble_boxes import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
AUGMENT = True
RUN = 19 # 19
RUN2 = 19
loc = f'runs/detect/train{RUN}'
loc2 = f'runs/detect/train{RUN2}'
SZ = (928, 448)
SZ2 = (928, 448) # 928

THRESH = 0.5
IOU_THRESH = 0.2
CONF = 0.5 # 0.5
YOLO_IOU_THR = 0.35
EXT = 'png'

model_path = f'{loc}/weights/best.pt'
model_path2 = f'{loc2}/weights/best.pt'
test_dir  = "/home/mutonkol/ML/KuyeserA/test_images/Images"
tif_dir = "/home/mutonkol/ML/KuyeserA/test_images/Images"
imgs = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(f'_pre_disaster.{EXT}')]
imgs_post = [f.replace('_pre_disaster', '_post_disaster') for f in imgs]
tifs = [os.path.join(tif_dir, f) for f in os.listdir(tif_dir) if f.endswith(f'_pre_disaster.{EXT}')]

def get_areas(boxes):
    areas = []
    for b in boxes:
        arr = abs(b[2] - b[0]) * abs(b[3] - b[1])
        areas.append(arr)
    return areas

def get_results(res, threshold=0.5):
    res     = res.cpu().numpy()
    cls     = res.boxes.cls
    conf    = res.boxes.conf
    boxes   = res.boxes.xyxyn#.tolist()
    mask    = conf > threshold
    conf = conf[mask]
    cls = cls[mask]
    boxes = boxes[mask]

    return cls, boxes, conf

def run_nms(bboxes, confs,labels, image_size, iou_thr=0.50, skip_box_thr=0.0001, weights=None):
    boxes =  [bbox for bbox in bboxes]
    scores = [conf for conf in confs]
    #labels = [np.ones(conf.shape[0]) for conf in confs]
    boxes, scores, labels = nms(boxes, scores, labels, weights=weights, iou_thr=iou_thr)
    #boxes = boxes*(image_size-1)
    return boxes, scores, labels

def run_nmw(bboxes, confs,labels, image_size, iou_thr=0.50, skip_box_thr=0.0001, weights=None):
    boxes =  [bbox for bbox in bboxes]
    scores = [conf for conf in confs]
    #labels = [np.ones(conf.shape[0]) for conf in confs]
    boxes, scores, labels = non_maximum_weighted(boxes, scores, labels, weights=weights, iou_thr=iou_thr)
    #boxes = boxes*(image_size-1)
    return boxes, scores, labels

def run_soft_nms(bboxes, confs,labels, image_size, iou_thr=0.50, skip_box_thr=0.0001,sigma=0.1, weights=None):
    boxes =  [bbox for bbox in bboxes]
    scores = [conf for conf in confs]
    #labels = [np.ones(conf.shape[0]) for conf in confs]
    boxes, scores, labels = soft_nms(boxes, scores, labels, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
    #boxes = boxes*(image_size-1)
    return boxes, scores, labels

def run_wbf(bboxes, confs,labels, image_size, iou_thr=0.50, skip_box_thr=0.0001, weights=None):
    #boxes =  [bbox/(image_size-1) for bbox in bboxes]
    boxes =  [bbox for bbox in bboxes]
    scores = [conf for conf in confs]
    #labels = [np.ones(conf.shape[0]) for conf in confs]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    #boxes = boxes*(image_size-1)
    return boxes, scores, labels


idx = 0
print(imgs[idx])
print(imgs_post[idx])
model = YOLO(model_path)
model2 = YOLO(model_path2)
results = model(imgs[idx], imgsz=SZ, augment=AUGMENT, conf=CONF)
img = results[0].plot()
IM = Image.fromarray(img)
IM.save('yolo.jpg')

results = model(imgs_post[idx], imgsz=SZ, augment=AUGMENT, conf=CONF)
img = results[0].plot()
IM = Image.fromarray(img)
IM.save('yolo_post.jpg')

#sys.exit(0)
img_files = []
preds = []
rep_str = {
        1: "_X_no_damage",
        2: "_X_minor_damage",
        3: "_X_major_damage",
        4: "_X_destroyed",
    }
os.makedirs("crops", exist_ok=True)
data = {}
for im in imgs:
    print(".", end="", flush=True)
    id = os.path.basename(im).replace(f'_pre_disaster.{EXT}', '')
    data[id] = {'pre':[], 'post':[]}
    im2 = im.replace('_pre_disaster', '_post_disaster')
    img_pre = cv2.imread(im)
    img_post = cv2.imread(im2)
    results_pre = model(im, imgsz=SZ, augment=AUGMENT, conf=CONF, iou=YOLO_IOU_THR)
    results_pre2 = model2(im, imgsz=SZ2, augment=AUGMENT, conf=CONF, iou=YOLO_IOU_THR)
    #results_post = model(im2, imgsz=SZ, augment=AUGMENT, conf=0.5)
    #results_post2 = model2(im2, imgsz=SZ, augment=AUGMENT, conf=0.5)
    for res_pre, res_pre2 in zip(results_pre, results_pre2):
        ccls_pre, bbxs_pre, cnfs_pre = get_results(res_pre, threshold=THRESH)
        ccls_pre2, bbxs_pre2, cnfs_pre2 = get_results(res_pre2, threshold=THRESH)
        boxes_pre, scores_pre, labels_pre = run_wbf(
                                     [bbxs_pre, bbxs_pre2], [cnfs_pre, cnfs_pre2],
                                     [ccls_pre, ccls_pre2],
                                     SZ[0], iou_thr=IOU_THRESH)
        #boxes_pre, scores_pre, labels_pre = run_wbf(
        #                             [bbxs_pre], [cnfs_pre], [ccls_pre],
        #                             SZ[0], iou_thr=IOU_THRESH)
        boxes_pre[:,0] = boxes_pre[:,0] * res_pre.orig_shape[1]
        boxes_pre[:,1] = boxes_pre[:,1] * res_pre.orig_shape[0]
        boxes_pre[:,2] = boxes_pre[:,2] * res_pre.orig_shape[1]
        boxes_pre[:,3] = boxes_pre[:,3] * res_pre.orig_shape[0]
        for i in range(len(boxes_pre)):
            x1,y1,x2,y2 = boxes_pre[i]
            crop_pre = img_pre[int(y1):int(y2), int(x1):int(x2)]
            crop_post = img_post[int(y1):int(y2), int(x1):int(x2)]
            if crop_pre.shape[0] != crop_post.shape[0]:
                height = min(crop_pre.shape[0], crop_post.shape[0])
                crop_pre = cv2.resize(crop_pre, (crop_pre.shape[1], height))
                crop_post = cv2.resize(crop_post, (crop_post.shape[1], height))
            # Stitch images side by side
            #stitched_image = np.hstack((crop_pre, crop_post))
            #cv2.imwrite(f"crops/{id}_X_{i}.jpg", stitched_image)
            #data[id]['pre'].append(f"{id}_X_{i}.jpg")
            cv2.imwrite(f"crops/{id}_X_{i}_pre.jpg", crop_pre)
            cv2.imwrite(f"crops/{id}_X_{i}_post.jpg", crop_post)
            data[id]['pre'].append(f"{id}_X_{i}_pre.jpg")
            data[id]['post'].append(f"{id}_X_{i}_post.jpg")

with open('data.pkl', 'wb') as f:
    pickle.dump(data, f)