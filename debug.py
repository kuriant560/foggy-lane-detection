import cv2
import numpy as np
import matplotlib.pyplot as plt

mask = cv2.imread("tusimple/train_set/seg_label/0313-1/10000/20.png", 0)

# Apply the same logic from the dataset_loader.py to verify it works
mask = (mask > 0).astype(np.float32)
mask = mask * 255.0

print("Min:", np.min(mask))
print("Max:", np.max(mask))

plt.imshow(mask, cmap='gray', vmin=0, vmax=255)
plt.title("Lane Mask")
plt.show()