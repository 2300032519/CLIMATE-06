import os
import sys
import glob
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# ======================================
# IMPORT MODEL
# ======================================

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.unet import UNet

# ======================================
# PATHS
# ======================================

DATASET_PATH = r"C:\CapstoneProject\Capstone\dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")

MODEL_PATH = r"C:\CapstoneProject\Capstone\outputs\best_model.pth"

# ======================================
# LOAD TRAINED MODEL
# ======================================

device = torch.device("cpu")

model = UNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ======================================
# LOAD IMAGE & MASK
# ======================================

image_files = sorted(glob.glob(os.path.join(TRAIN_PATH, "*_sat.jpg")))
mask_files = sorted(glob.glob(os.path.join(TRAIN_PATH, "*_mask.png")))

print("Training Images :", len(image_files))
print("Training Masks  :", len(mask_files))

image = cv2.imread(image_files[0])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mask = cv2.imread(mask_files[0], cv2.IMREAD_GRAYSCALE)

original = image.copy()

# ======================================
# PREPROCESS
# ======================================

image = cv2.resize(image, (256, 256))
mask = cv2.resize(mask, (256, 256))

image = image.astype(np.float32) / 255.0
mask = (mask > 127).astype(np.uint8)

image_tensor = torch.tensor(
    np.transpose(image, (2, 0, 1)),
    dtype=torch.float32
).unsqueeze(0)

# ======================================
# PREDICTION
# ======================================
name
with torch.no_grad():
    prediction = model(image_tensor)
    prediction = torch.sigmoid(prediction)

prediction = prediction.squeeze().numpy()

print("Min Probability :", prediction.min())
print("Max Probability :", prediction.max())

# Review-II visualization threshold
# Current
binary_prediction = (prediction > 0.5).astype(np.uint8)

# Better for your trained model
#binary_prediction = (prediction > 0.15).astype(np.uint8)

# ======================================
# DISPLAY RESULTS
# ======================================

plt.figure(figsize=(18, 5))

plt.subplot(1, 4, 1)
plt.imshow(original)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(mask, cmap="gray")
plt.title("Ground Truth")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(prediction, cmap="viridis")
plt.title("Probability Map")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(binary_prediction, cmap="gray")
plt.title("Predicted Mask")
plt.axis("off")

plt.tight_layout()
plt.show()