import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from dataset_loader import LaneDataset
from bdd_loader import BDD100kLaneDataset
from model.unet import UNet

def main():
    # 1. Base Configuration
    epochs = 10
    batch_size = 4
    learning_rate = 0.001
    
    # Pointing exactly to the TuSimple dataset structure we built
    data_dir = "tusimple/train_set"
    json_file = "tusimple/train_set/label_data_0313.json"
    mask_dir = "tusimple/train_set/seg_label"
    
    # Enable hardware acceleration (CUDA for Windows/Linux, MPS for macOS Apple Silicon)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Hardware Acceleration: Using {device}\n")

    # 2. Load Dataset into DataLoader
    print("Loading TuSimple Dataset...")
    # Passing the exact kwargs we standardized earlier
    tusimple_dataset = LaneDataset(data_dir=data_dir, json_file=json_file, mask_dir=mask_dir, apply_fog_aug=True)
    
    print("Loading BDD100k Dataset...")
    bdd_images_dir = "bdd100k_data/bdd100k/bdd100k/images/100k/train"
    bdd_labels_file = "bdd100k_data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
    bdd_dataset = BDD100kLaneDataset(images_dir=bdd_images_dir, labels_path=bdd_labels_file)

    print("Merging Datasets mathematically...")
    combined_dataset = ConcatDataset([tusimple_dataset, bdd_dataset])
    
    # Reduced batch_size or num_workers might be needed on macs reading 100k items, but 4 is safe
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    print(f"Success! {len(combined_dataset)} items loaded into DataLoader.\n")

    # 3. Model Architecture
    # The U-Net requires 3 input channels (RGB) and produces 1 output channel (Binary Mask)
    # We pass these kwargs out of convention, although they need to be implemented in unet.py
    model = UNet() 
    model = model.to(device)

    # Check for existing weights to resume training
    start_epoch = 0
    weight_files = glob.glob("weights/model_epoch_*.pth")
    if weight_files:
        latest_model = max(weight_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        start_epoch = int(latest_model.split('_')[-1].split('.')[0])
        print(f"Resuming training from {latest_model} (Starting at Epoch {start_epoch + 1})")
        model.load_state_dict(torch.load(latest_model, map_location=device, weights_only=True))

    # 4. Define mathematical Loss Function and Optimizer
    # We apply a pos_weight of 25.0 to heavily penalize missing the rare white lane pixels!
    # This mathematically completely prevents the network from collapsing into a "predict trees" lazy state.
    pos_weight_tensor = torch.tensor([25.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 5. Core Training Loop
    print("Starting Training Loop...")
    for epoch in range(start_epoch, epochs):
        model.train() # Locks standard layers like Dropout into active training states
        current_epoch_loss = 0.0
        
        # Iterate over batches
        for batch_idx, (images, masks) in enumerate(dataloader):
            # Send data physical bounds to the active GPU/Hardware device
            images = images.to(device)
            # IMPORTANT: In dataset_loader, masks are scaled to 255 for visibility!
            # BCEWithLogitsLoss algorithm requires ground truths specifically strictly between [0.0, 1.0].
            # Therefore, we dynamically scale them back down.
            masks = (masks / 255.0).to(device)
            
            # Zero-out the historical gradients left over from the previous iteration
            optimizer.zero_grad()
            
            # Execute step forward (generate predictions)
            predictions = model(images)
            
            # Mathematically compute the penalty/loss difference against the real mask
            loss = criterion(predictions, masks)
            
            # Execute step backwards (propagate penalty calculation gradients)
            loss.backward()
            
            # Move the optimizer logic forward (apply updates to the model weights)
            optimizer.step()
            
            current_epoch_loss += loss.item()
            
            # Print batch diagnostics so we know the process isn't frozen!
            if batch_idx % 50 == 0:
                print(f"   -> Batch [{batch_idx}/{len(dataloader)}] | Current Loss: {loss.item():.5f}")
                
        # 6. Print diagnostics
        avg_loss = current_epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | Average Loss: {avg_loss:.5f}")

        # Save model after each epoch
        os.makedirs("weights", exist_ok=True)
        epoch_save_path = f"weights/model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), epoch_save_path)
        print(f"Model saved for epoch {epoch+1} to: {epoch_save_path}")

    # 7. Save Model Weights
    print("\nTraining Finished!")
    save_path = "weights/model_final.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Final model state successfully saved to: {save_path}")

if __name__ == "__main__":
    main()
