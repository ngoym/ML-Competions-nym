import os
import cv2
import pickle
from timm import create_model
import pdb
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import pandas as pd

SZ = 128
FOLD = 0
MODEL = 'efficientvit_b0.r224_in1k'

DATA_DIR = "/home/mutonkol/ML/KuyeserA/code/basic"
CROPS_DIR = "crops" # "mmde_crops" for mmdet, "crops" for yolo
disaster_dir = os.path.join(DATA_DIR, CROPS_DIR)
DATA_FILE = "data.pkl" # "mmdet_data.pkl" for mmdet, data.pkl for yolo
with open(DATA_FILE, "rb") as f:
    data = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_df = pd.read_csv("test_ndvi.csv")

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

valid_transform = A.Compose([
    A.Resize(SZ, SZ),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], additional_targets={'image2': 'image'})


model = DisasterClassifier()
model.load_state_dict(torch.load(f"efficientvit_b0.r224_in1k_3_patch_model_fold_0.pth"))
model = model.to(device)
model.eval()
ids = []
damage = []
d_count = 0
for i in data.keys():
    print("[INFO] - Processing", i, flush=True)
    d = {
        0: {'id':f"{i}_X_no_damage", 'count':0},
        1: {'id':f"{i}_X_minor_damage", 'count':0},
        2: {'id':f"{i}_X_major_damage", 'count':0},
        3: {'id':f"{i}_X_destroyed", 'count':0}
    }
    nvdi = torch.tensor(test_df[test_df['id'] == i]['NDVI_mean'].values[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    for j in data[i]['pre']:
        pre_image = cv2.imread(os.path.join(disaster_dir, j))
        post_image = cv2.imread(os.path.join(disaster_dir, j.replace("_pre.jpg", "_post.jpg")))
        pre_size = os.path.getsize(os.path.join(disaster_dir, j))
        post_size = os.path.getsize(os.path.join(disaster_dir, j.replace("_pre.jpg", "_post.jpg")))

        #if pre_size < 1024*2 or post_size < 1024*2:
            #if pre_size/post_size >= 2 or post_size < pre_size:
            #    d[3]['count'] += 1
            #elif post_size > pre_size:
            #    d[0]['count'] += 1
        #    continue

        pre_image = cv2.cvtColor(pre_image, cv2.COLOR_BGR2RGB)
        post_image = cv2.cvtColor(post_image, cv2.COLOR_BGR2RGB)
        pre_image = valid_transform(image=pre_image)['image']
        post_image = valid_transform(image=post_image)['image']
        pre_image_ = pre_image.unsqueeze(0).to(device)
        post_image_ = post_image.unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(pre_image_, post_image_, nvdi)
        if out.item() > 0.5:
            d[3]['count'] += 1
            d_count += 1
            #cv2.imwrite("abs_pre_image.png", pre_image.permute(1,2,0).cpu().numpy())
            #cv2.imwrite("abs_post_image.png", post_image.permute(1,2,0).cpu().numpy())
            #db.set_trace()
        else:
            d[0]['count'] += 1

    for k in d.keys():
        ids.append(d[k]['id'])
        damage.append(d[k]['count'])

submission = pd.DataFrame({'id':ids, 'damage':damage})
submission.to_csv(f"cls_{MODEL}_submission.csv", index=False)
print("[INFO] - Num destroyed", d_count, flush=True)
