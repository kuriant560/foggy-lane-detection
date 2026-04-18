import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class LaneDataset(Dataset):
    """
    A PyTorch Dataset for loading lane detection images and matching pre-rendered segmentation masks.
    """
    
    def __init__(self, data_dir, json_file, mask_dir=None, apply_fog_aug=False, img_size=(256, 512)):
        """
        Args:
            data_dir (str): Base directory containing the images.
            json_file (str): Path to the JSON annotation file (e.g., label_data_*.json).
            mask_dir (str): Directory where segmentation masks are stored. If None, assumes masks 
                            are next to images or path can be inferred.
            apply_fog_aug (bool): Whether to apply foggy augmentations using Albumentations.
            img_size (tuple): Target size for resizing the images (height, width).
        """
        self.data_dir = data_dir
        self.mask_dir = mask_dir if mask_dir else data_dir
        self.img_size = img_size
        self.data = []
        
        # Load the JSON file line by line
        with open(json_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
                
        # Base Transformation: Just resizing and converting to PyTorch tensors
        self.base_transform = A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            ToTensorV2()
        ])
        
        # Fog Transformation: Adds adverse weather conditions to make the model robust
        self.fog_transform = A.Compose([
            A.RandomFog(p=0.7),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Resize(height=img_size[0], width=img_size[1]),
            ToTensorV2() 
        ])
        
        self.transform = self.fog_transform if apply_fog_aug else self.base_transform

    def __len__(self):
        """Returns the total number of images/samples in the loaded JSON file."""
        return len(self.data)

    def __getitem__(self, idx):
        """Loads the image and its corresponding segmentation mask from disk."""
        item = self.data[idx]
        image_path = os.path.join(self.data_dir, item['raw_file'])
        
        # 1. Load the original image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image could not be found at {image_path}")
            
        # OpenCV loads in BGR; neural nets generally expect RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Derive mask path
        # The TuSimple masks are typically in 'seg_label/' mirroring the 'clips/' structure.
        # So 'clips/0313-1/10000/20.jpg' becomes '0313-1/10000/20.png'
        mask_filename = item.get('mask_file', item['raw_file'].replace('.jpg', '.png').replace('clips/', ''))
        mask_path = os.path.join(self.mask_dir, mask_filename)
        
        # 3. Load mask as grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask could not be found at {mask_path}")
            
        # 4. Convert mask to binary and expand to 255 for visibility (Lane = 255, Background = 0)
        mask = (mask > 0).astype(np.float32)
        mask = mask * 255.0
        
        # 5. Apply Albumentations (resizing and augmentations) to BOTH image and mask identically
        augmented = self.transform(image=image, mask=mask)
        image_tensor = augmented['image']
        mask_tensor = augmented['mask']
        
        # 6. Standardize data types and dimensions
        # Image tensors: Float normalized to [0, 1] range
        image_tensor = image_tensor.float() / 255.0
        
        # Ensure mask is converted to float identically and has 1 channel (1, H, W)
        mask_tensor = mask_tensor.clone().detach().float()
        
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
            
        return image_tensor, mask_tensor
