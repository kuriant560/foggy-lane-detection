import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BDD100kLaneDataset(Dataset):
    def __init__(self, images_dir, labels_path, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        
        print("Loading BDD100k JSON (this may take a moment)...")
        with open(labels_path, 'r') as f:
            raw_data = json.load(f)
            
        # Extract only the necessary polygon coordinates to save memory!
        self.data = []
        limit = 30000  # Safe limit for 8GB RAM and ~4 hour training
        for item in raw_data:
            if len(self.data) >= limit:
                break
                
            lane_polys = []
            if 'labels' in item:
                for label in item['labels']:
                    if label['category'] == 'lane':
                        if 'poly2d' in label:
                            for poly in label['poly2d']:
                                lane_polys.append(poly['vertices'])
            
            # Only store images that actively have lane boundaries
            if len(lane_polys) > 0:
                self.data.append({
                    'name': item['name'],
                    'polys': lane_polys
                })

        print(f"Success! Loaded {len(self.data)} BDD100k items into DataLoader.")

        if self.transform is None:
            self.transform = A.Compose([
                A.Resize(256, 512),
                A.RandomFog(p=0.7), # Synthetic weather rendering!
                A.RandomBrightnessContrast(p=0.5),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_name = item['name']
        img_path = os.path.join(self.images_dir, img_name)
        
        # Kaggle dataset routing logic (images are sometimes randomly subsetted to bypass upload caps)
        if not os.path.exists(img_path):
            candidates = ['trainA', 'trainB', 'testA', 'testB', 'val']
            for c in candidates:
                fallback_path = os.path.join(self.images_dir, c, img_name)
                if os.path.exists(fallback_path):
                    img_path = fallback_path
                    break
        
        # Original BDD100k resolution is exactly 1280x720
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create zero mask background
        mask = np.zeros((720, 1280), dtype=np.float32)
        
        # Draw the literal geometric poly lines directly into the OpenCV byte array
        for poly in item['polys']:
            pts = np.array(poly, np.int32)
            pts = pts.reshape((-1, 1, 2))
            # Color must be 255 so train.py successfully scales it down by 255.0 to 1.0
            cv2.polylines(mask, [pts], isClosed=False, color=255.0, thickness=8)
            
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image_tensor = augmented['image']
            mask_tensor = augmented['mask']
            
        # Match TuSimple normalization explicitly!
        image_tensor = image_tensor.float() / 255.0
        mask_tensor = mask_tensor.clone().detach().float()
        
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
            
        return image_tensor, mask_tensor
