import matplotlib.pyplot as plt
from dataset_loader import LaneDataset

def main():
    # Setup dataset paths according to TuSimple's standard folder structure
    data_dir = "tusimple/train_set"
    json_file = "tusimple/train_set/label_data_0313.json"
    mask_dir = "tusimple/train_set/seg_label"
    
    print("Testing LaneDataset initialization...")
    # Disabling fog augmentation for now to ensure we can clearly see the base mask
    dataset = LaneDataset(data_dir=data_dir, json_file=json_file, mask_dir=mask_dir, apply_fog_aug=False)
    print(f"Success! Loaded {len(dataset)} pairs from {json_file}")
    
    if len(dataset) == 0:
        print("Dataset is empty. Check your paths!")
        return

    print("\nFetching sample 0...")
    img_tensor, mask_tensor = dataset[0]
    
    print(f">> Image Tensor: shape={img_tensor.shape}, dtype={img_tensor.dtype}, max={img_tensor.max():.2f}")
    print(f">> Mask Tensor:  shape={mask_tensor.shape}, dtype={mask_tensor.dtype}, min={mask_tensor.min():.0f}, max={mask_tensor.max():.0f}")
    
    # Squeeze the single channel dimension for visualization (C, H, W) -> (H, W)
    mask_img = mask_tensor.squeeze(0).numpy()
    
    # Permute the image for Matplotlib (C, H, W) -> (H, W, C)
    rgb_img = img_tensor.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].imshow(rgb_img)
    axes[0].set_title(f"Augmented Image (256x512)")
    axes[0].axis("off")
    
    # Display the mask enforcing strict 0-255 viewing limits
    axes[1].imshow(mask_img, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title("Processed Tensor Mask (0 = Black, 255 = White)")
    axes[1].axis("off")
    
    print("\nDisplaying visualization window...")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
