import os
import sys
import glob
import cv2
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.unet import UNet

# ---------------- PATHS ----------------

DATASET_PATH = r"C:\CapstoneProject\Capstone\dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
MODEL_PATH = r"C:\CapstoneProject\Capstone\outputs\best_model.pth"

# ---------------- LOAD MODEL ----------------

device = torch.device("cpu")

model = UNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ---------------- LOAD IMAGE ----------------

image_path = sorted(glob.glob(os.path.join(TRAIN_PATH, "*_sat.jpg")))[0]
mask_path = sorted(glob.glob(os.path.join(TRAIN_PATH, "*_mask.png")))[0]

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

image = cv2.resize(image, (256,256))
mask = cv2.resize(mask, (256,256))

image = image.astype(np.float32)/255.0
mask = (mask > 127).astype(np.uint8)

tensor = torch.tensor(
    np.transpose(image,(2,0,1)),
    dtype=torch.float32
).unsqueeze(0)

# ---------------- PREDICT ----------------

with torch.no_grad():
    pred = torch.sigmoid(model(tensor))

pred = pred.squeeze().numpy()
pred = (pred > 0.15).astype(np.uint8)

# ---------------- METRICS ----------------

intersection = np.logical_and(mask, pred).sum()
union = np.logical_or(mask, pred).sum()

iou = intersection / (union + 1e-8)

dice = (2 * intersection) / (mask.sum() + pred.sum() + 1e-8)

precision = intersection / (pred.sum() + 1e-8)

recall = intersection / (mask.sum() + 1e-8)

print("\nEvaluation Metrics")
print("-----------------------")
print(f"IoU Score      : {iou:.4f}")
print(f"Dice Score     : {dice:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")