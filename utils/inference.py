import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model.unet import UNet

class LaneDetector:
    """
    A standalone Inference Engine wrapper that bridges the trained Pytorch weights
    back into a format compatible with front-end GUI display applications.
    """
    def __init__(self, model_path="weights/model.pth", device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
            
        print(f"Loading U-Net inference engine on: {self.device}")
        
        self.model = UNet()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval() # Hard-lock the model to prevent accidental math updates
        
        # The inference inputs MUST conform exactly to the mathematical structure it trained on
        self.transform = A.Compose([
            A.Resize(height=256, width=512),
            ToTensorV2()
        ])

    def predict(self, image_bgr):
        # 1. Store original dimensions so we can seamlessly stretch the mask back later!
        original_h, original_w = image_bgr.shape[:2]
        
        # 2. BGR to RGB conversion
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # 3. Transform to tensor format [0.0, 1.0] identical to dataset_loader.py
        augmented = self.transform(image=image_rgb)
        tensor = augmented['image'].float() / 255.0
        
        # Neural nets strictly expect batches: (C, H, W) -> (B, C, H, W) where B=1
        tensor = tensor.unsqueeze(0).to(self.device) 
        
        # 4. Bypass gradient math completely to drastically slash latency
        with torch.no_grad():
            output = self.model(tensor)
            # The model spits out raw unconstrained numbers; Sigmoid clamps them exclusively between 0% and 100% confidence
            prob_mask = torch.sigmoid(output)
            
        # 5. Bring data out of the GPU back to system memory (CPU -> Numpy)
        prob_mask = prob_mask.squeeze().cpu().numpy()
        
        # 6. Magically enlarge the 256x512 continuous float probability mask perfectly back to the HD aspect ratio
        prob_mask_resized = cv2.resize(prob_mask, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        return prob_mask_resized

def overlay_unet_mask(image_bgr, binary_mask, color=(0, 255, 0), alpha=0.5):
    """
    Applies the mathematical mask array seamlessly onto the raw HD image.
    Uses an Alpha blend so the glowing line doesn't destroy the asphalt texture beneath it.
    """
    overlay = image_bgr.copy()
    colored_mask = np.zeros_like(image_bgr)
    
    # Broadcast the color array across all locations the UNet tagged as "Lane"
    colored_mask[binary_mask == 1] = color
    
    # Filter matrix indices
    mask_indices = binary_mask == 1
    
    # Perform additive glow blending directly at those indices
    overlay[mask_indices] = cv2.addWeighted(image_bgr, 1 - alpha, colored_mask, alpha, 0)[mask_indices]
    
    return overlay
