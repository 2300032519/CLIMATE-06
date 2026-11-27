import os
import glob
import cv2
import matplotlib.pyplot as plt

# Dataset path
DATASET_PATH = r"C:\CapstoneProject\Capstone\dataset"

TRAIN_PATH = os.path.join(DATASET_PATH, "train")

# Read satellite images and masks
image_files = sorted(glob.glob(os.path.join(TRAIN_PATH, "*_sat.jpg")))
mask_files = sorted(glob.glob(os.path.join(TRAIN_PATH, "*_mask.png")))

print("Satellite Images :", len(image_files))
print("Road Masks       :", len(mask_files))

# Load first sample
image = cv2.imread(image_files[0])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mask = cv2.imread(mask_files[0], cv2.IMREAD_GRAYSCALE)

print("Image Shape :", image.shape)
print("Mask Shape  :", mask.shape)

# Display
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Satellite Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(mask, cmap="gray")
plt.title("Road Mask")
plt.axis("off")

plt.tight_layout()
plt.show()