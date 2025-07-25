# Description

Using satellite imagery and the xBD dataset, we identified cyclone-damaged houses in Blantyre, Malawi. A critical step toward improving post-disaster recovery and equity-focused AI interventions.


Competed with [Kolesh](https://github.com/koleshjr) on this and won third place.

# Team central_park solution
## General Description
Our solution consists of a two stage approach: object detection to detect buildings in pre-disaster images and image classification, comparing patches and pre- and post-disaster images. The classification part makes use of ASDI datasets.

## Methods
### Data preparation
- Combine tier1 and tier3 images and labels into a common directory.
- Run `prep.py` to convert the provided *.tif files to *.png. Take care to adjust the `TEST` flag and set the paths correctly.

### Object detection
#### Training
Train Yolo11l model on pre-disaster images only from the tier1 and tier3 xview2 dataset. We used "test" as validation data.

The aim here is to train a model that is able to detect buldings in an image. This is a single class detection problem with configuration
```
names:
- building
num_classes: 1
train: /home/ML/KuyeserA/xview2/geotiffs/train/images
val: /home/ML/KuyeserA/xview2/geotiffs/test/images
```
and
```
EPS = 60
BS = 16
IMZ = 800
MOSAIC = 1.0
device = [1]
MODEL = 'yolo11l.pt'
model = YOLO(MODEL)
model.train(
    data="data.yaml", epochs=EPS, imgsz=IMZ, \
    device=device, batch=BS, plots=True, \
    flipud=0.3,mixup=0.5, \
    erasing=0.4,copy_paste=0.0, \
    cos_lr=True, \
    hsv_s=0.3, hsv_v=0.3, \
    mosaic=MOSAIC, \
    close_mosaic=20, \
)
```
Data preparation is done in `yolo_prep.py`, training is done in `yolo.py`.

This model achieves an F1 score of 0.72. We provide the weights in yolo_run19.zip.

#### Inference
Inference was done at a resolution of `(928, 448)` using WBF ensemble with thresholds found in `yolo_test.py`. Essentially inference is done on pre-disaster images only and then coordinates of found buildings are used to create crops in pre-  and post-disaster images.

### Classification
- The first step here is to regenerate yolo labels from the detection stage, by replacing `cat = 0` in `Line 151` of `yolo_prep.py` with `cat = labels[i]` in order to recover the different building damage classes. Once this is done, run `patchify.py` to extract pre- and post-disaster patch pairs containing a single building. This will create a directory with the patches as well as a `train.csv` file.

- Next, we run `gen_data.py` to generate NDVI from ASDI data. This step generates `train_s2.csv` and `test_df.csv` files.

- We then run `compute_bands.py` to compute NDVI for test data from ASDI data and merge `train_s2.csv` and `train.csv` from the previous two steps.

- We then can train a classifier for different classes. We found that training classifiers for `minor_damage` and `major_damage` were not accurate enough with AUC scores of ~0.8, and the vast distribution shift between xview2 and test data. We therefore only considered a classifier for the `destroyed` class which had an AUC score of ~0.93 in training. We used a Siamese neural network where the pre- and post-disaster images share the same weights. We used `efficientvit_b0.r224_in1k` from `timm` as the backbone. The model is given below
```
class DisasterClassifier(nn.Module):
    def __init__(self):
        super(DisasterClassifier, self).__init__()
        self.features = create_model(MODEL, pretrained=True, num_classes=0)

        self.fc = nn.Linear(1+1, 1)

    def euclidean_distance(self, x1, x2):
        return torch.sqrt(torch.sum((x1 - x2)**2, dim=1, keepdim=True))

    def forward(self, pre_image, post_image, x):
        # Extract features from both images
        pre_features = self.features(pre_image)
        post_features = self.features(post_image)

        distance = self.euclidean_distance(pre_features, post_features)
        distance = torch.cat([distance, x], dim=1)
        out = torch.sigmoid(self.fc(distance))
        return out
```
`x` in the forward path is the post-disaster NVDI metric from ASDI datasets. The model was trained for 50 epochs.

Training details are in `siam.py`.

### Generating submission
We use the "destroyed" building classifier to check whether the building has been destroyed, otherwise we classify the build as "no damage". We essentially use crops generated from the yolo inference stage to feed the classifier and generate the submission *.csv file. Details are in `predict_cls.py`.

Furthermore, we include a notebook, `inference-notebook.ipynb`, that can be run end-to-end to generate the submission file. We have additionally included the test images, converted to *.png format, so that the  inference may work almost effortlessly - one just has to change the paths to the images and model weights (also included).
