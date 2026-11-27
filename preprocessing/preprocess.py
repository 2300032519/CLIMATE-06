import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# DATASET PATH
# =====================================
DATASET_PATH = r"C:\CapstoneProject\Capstone\dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")

# =====================================
# LOAD FIRST IMAGE & MASK
# =====================================
image_path = sorted(glob.glob(os.path.join(TRAIN_PATH, "*_sat.jpg")))[0]
mask_path = sorted(glob.glob(os.path.join(TRAIN_PATH, "*_mask.png")))[0]

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# =====================================
# PREPROCESSING
# =====================================

# Resize to 256 × 256
image = cv2.resize(image, (256, 256))
mask = cv2.resize(mask, (256, 256))

# Normalize image (0–1)
image = image.astype(np.float32) / 255.0

# Convert mask to binary
mask = (mask > 127).astype(np.uint8)

# =====================================
# VERIFY
# =====================================
print("Image Shape :", image.shape)
print("Mask Shape  :", mask.shape)
print("Pixel Range :", image.min(), "to", image.max())
print("Mask Values :", np.unique(mask))

# =====================================
# DISPLAY
# =====================================
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Preprocessed Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="gray")
plt.title("Binary Road Mask")
plt.axis("off")

plt.tight_layout()
plt.show()