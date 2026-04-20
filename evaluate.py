import os
import glob
import torch
from torch.utils.data import DataLoader
from bdd_loader import BDD100kLaneDataset
from model.unet import UNet

def calculate_metrics(pred_mask, true_mask):
    """
    Calculates Intersection over Union (IoU) and F1-Score (Dice Coefficient)
    between a binary prediction mask and a binary ground-truth mask.
    """
    # Flatten the 2D matrices into 1D arrays for easy mathematical comparison
    pred = pred_mask.view(-1)
    target = true_mask.view(-1)
    
    # True Positives (Where both predicted and true are 1)
    intersection = (pred * target).sum()
    
    # All positive instances in both arrays
    pred_sum = pred.sum()
    target_sum = target.sum()
    
    # Union (Area of overlap)
    union = pred_sum + target_sum - intersection
    
    # We add 1e-6 (a tiny number) to prevent division by zero errors if an image has zero lanes
    iou = (intersection + 1e-6) / (union + 1e-6)
    f1 = (2. * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)
    
    return iou.item(), f1.item()

def main():
    # 1. Device Setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Executing mathematical evaluation on: {device}")

    # 2. Dynamically locate the latest weights
    weight_files = glob.glob("weights/model_epoch_*.pth")
    if not weight_files:
        print("ERROR: No trained weights found in the 'weights/' directory. Please run train.py first.")
        return
        
    latest_model_path = max(weight_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(f"Loading weights from: {latest_model_path}")
    
    # 3. Load Model
    model = UNet()
    model.load_state_dict(torch.load(latest_model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval() # Disable dropout and batch norm tracking for pure evaluation

    # 4. Load the Validation Dataset (Data the model has NEVER seen during training)
    print("Loading BDD100k Validation Dataset...")
    val_images_dir = "bdd100k_data/bdd100k/bdd100k/images/100k/val"
    val_labels_file = "bdd100k_data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"
    
    val_dataset = BDD100kLaneDataset(images_dir=val_images_dir, labels_path=val_labels_file)
    dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    print(f"Found {len(val_dataset)} validation images. Beginning evaluation...")

    total_iou = 0.0
    total_f1 = 0.0
    num_batches = len(dataloader)

    # 5. Core Evaluation Loop
    with torch.no_grad(): # Completely bypass gradient tracking to save massive amounts of RAM
        for batch_idx, (images, true_masks) in enumerate(dataloader):
            images = images.to(device)
            # The dataset loader scales masks to 255 for visibility. Scale back to exactly 0.0 or 1.0
            true_masks = (true_masks / 255.0).to(device)
            
            # Forward Pass
            logits = model(images)
            
            # Constrain raw logits to probabilities [0.0, 1.0] using Sigmoid
            probs = torch.sigmoid(logits)
            
            # Binarize predictions (If confidence > 50%, classify as "Lane Pixel")
            pred_masks = (probs > 0.5).float()
            
            # Calculate metrics
            iou, f1 = calculate_metrics(pred_masks, true_masks)
            
            total_iou += iou
            total_f1 += f1
            
            if batch_idx % 25 == 0:
                print(f"   -> Processing Batch [{batch_idx}/{num_batches}] | Current Batch F1-Score: {f1:.4f}")

    # 6. Final Results Compilation
    avg_iou = total_iou / num_batches
    avg_f1 = total_f1 / num_batches
    
    print("\n" + "="*50)
    print("🎯 FINAL EVALUATION METRICS")
    print("="*50)
    print(f"Model Evaluated:  {os.path.basename(latest_model_path)}")
    print(f"Dataset Size:     {len(val_dataset)} images (BDD100k Validation Set)")
    print(f"Average IoU:      {avg_iou:.4f}  <-- Intersection over Union")
    print(f"Average F1-Score: {avg_f1:.4f}  <-- Dice Coefficient")
    print("="*50)
    print("You can copy these numbers directly into your research paper's Results section.")

if __name__ == "__main__":
    main()
