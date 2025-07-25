import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import pandas as pd
import pdb
from tqdm import tqdm
from timm import create_model
import argparse

parser = argparse.ArgumentParser(description='Disaster Classifier')
parser.add_argument('--model', type=str, default='efficientvit_b0.r224_in1k', help='Model name')
parser.add_argument('--sz', type=int, default=128, help='Image size')
parser.add_argument('--fold', type=int, default=0, help='Fold')
parser.add_argument('--bs', type=int, default=256, help='Batch size')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--category', type=int, default=3, help='category to classify. 0 for no-damage, 1 for minor-damage, 2 for major-damage, 3 for destroyed')

args = parser.parse_args()

SEED = 42
SZ = args.sz
FOLD = args.fold
MODEL = args.model
LR = args.lr
NUM_EPOCHS = args.epochs
BS = args.bs
category = args.category

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

DATA_DIR = "/home/mutonkol/ML/KuyeserA/xview2/geotiffs"
pre_disaster_dir = os.path.join(DATA_DIR, "train", "pre_disaster")
post_disaster_dir = os.path.join(DATA_DIR, "train", "post_disaster")

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)
damage_class_to_id = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}

df = pd.read_csv("train_ndvi.csv")
if category != 3:
    df = df[df['label'] != 3].reset_index()
df['label'] = df['label'].apply(lambda x: 0 if x != category else x)
df['label'] = df['label'].apply(lambda x: 1 if x == category else x)

df['fold'] = -1
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
    df.loc[val_idx, 'fold'] = fold

class DisasterDataset(Dataset):
    def __init__(self, df, pre_disaster_dir, post_disaster_dir, transforms=None):
        self.df = df
        self.pre_disaster_dir = pre_disaster_dir
        self.post_disaster_dir = post_disaster_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pre_img_id = self.df.iloc[idx]['pre_image']
        post_img_id = self.df.iloc[idx]['post_image']
        label = torch.tensor(self.df.iloc[idx]['label'], dtype=torch.float32)
        NDVI_mean = torch.tensor(self.df.iloc[idx]['NDVI_mean'], dtype=torch.float32)
        pre_image_path = os.path.join(self.pre_disaster_dir, pre_img_id)
        post_image_path = os.path.join(self.post_disaster_dir, post_img_id)
        # Read images
        pre_image = cv2.imread(pre_image_path)
        post_image = cv2.imread(post_image_path)
        pre_image = cv2.cvtColor(pre_image, cv2.COLOR_BGR2RGB)
        post_image = cv2.cvtColor(post_image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=pre_image, image2=post_image)
            pre_image = augmented['image']
            post_image = augmented['image2']

        return pre_image, post_image, label, NDVI_mean


# Define Albumentations augmentations
train_transform = A.Compose([
    A.Resize(SZ, SZ),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(
        scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.1, border_mode=0
    ),
    #A.PadIfNeeded(min_height=320, min_width=320, always_apply=True),
    #A.RandomCrop(height=SZ, width=SZ, always_apply=True),
    A.GaussNoise(p=0.5),
    A.Perspective(p=0.3),
    A.OneOf(
        [
            A.CLAHE(p=1),
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1),
        ],
        p=0.7, #0.9
    ),
    A.OneOf(
        [
            A.Sharpen(p=1),
            A.Blur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),
        ],
        p=0.3,
    ),
    A.OneOf(
        [
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4, p=1.0),
            A.HueSaturationValue(p=1),
        ],
        p=0.5,
    ),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], additional_targets={'image2': 'image'})

valid_transform = A.Compose([
    A.Resize(SZ, SZ),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], additional_targets={'image2': 'image'})

df_train = df[df['fold'] != FOLD].reset_index(drop=True)
df_valid = df[df['fold'] == FOLD].reset_index(drop=True)

train_dataset = DisasterDataset(df_train, pre_disaster_dir, post_disaster_dir, train_transform)
valid_dataset = DisasterDataset(df_valid, pre_disaster_dir, post_disaster_dir, valid_transform)

train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(valid_dataset, batch_size=BS, shuffle=False, num_workers=8, pin_memory=True)

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

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=10, save_path=f'{MODEL}_best_patch_model.pth'):
    model = model.to(device)
    best_val_acc = 0.0
    best_val_auc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for pre_images, post_images, labels, NDVI_mean in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
            pre_images, post_images, labels, NDVI_mean = pre_images.to(device), post_images.to(device), labels.to(device), NDVI_mean.to(device)

            # Zero the gradient buffers
            optimizer.zero_grad()

            # Forward pass
            outputs = model(pre_images, post_images, NDVI_mean.unsqueeze(1))
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * pre_images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}, Training Loss: {epoch_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_running_accuracy = 0.0
        all_labels = []
        all_preds = []
        val_auc = 0.0

        with torch.no_grad():
            for pre_images, post_images, labels, NDVI_mean in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
                pre_images, post_images, labels, NDVI_mean = pre_images.to(device), post_images.to(device), labels.to(device), NDVI_mean.to(device)

                outputs = model(pre_images, post_images, NDVI_mean.unsqueeze(1))
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                # Collect the labels and predicted probabilities for AUC
                all_labels.append(labels.cpu().numpy())
                all_preds.append(outputs.cpu().numpy())

        val_epoch_loss = val_loss / len(val_loader.dataset)
        # Flatten the list of labels and predictions for AUC computation
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        val_epoch_auc = roc_auc_score(all_labels, all_preds, multi_class='ovr', average='macro')
        print(f"Epoch {epoch+1}, Validation Loss: {val_epoch_loss:.4f}, Validation AUC: {val_epoch_auc:.4f}")

        # Save the best model
        if val_epoch_auc > best_val_auc:
            best_val_auc = val_epoch_auc
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved AUC: {val_epoch_auc:.4f}")
        # Step the learning rate scheduler
        scheduler.step(val_epoch_loss)

# Instantiate the model
model = DisasterClassifier()

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

train_and_validate(
    model, train_loader, val_loader,
    criterion, optimizer, scheduler,
    device=device, num_epochs=NUM_EPOCHS,
    save_path=f'{MODEL}_{category}_patch_model_fold_{FOLD}.pth')