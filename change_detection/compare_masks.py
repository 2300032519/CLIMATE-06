import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

TRAIN_PATH = r"C:\CapstoneProject\Capstone\dataset\train"

# Load two different road masks
mask_files = sorted(glob.glob(os.path.join(TRAIN_PATH, "*_mask.png")))

old_mask = cv2.imread(mask_files[0], cv2.IMREAD_GRAYSCALE)
new_mask = cv2.imread(mask_files[1], cv2.IMREAD_GRAYSCALE)

old_mask = cv2.resize(old_mask, (256,256))
new_mask = cv2.resize(new_mask, (256,256))

old_mask = (old_mask > 127).astype(np.uint8)
new_mask = (new_mask > 127).astype(np.uint8)

# CHANGE DETECTION

new_roads = np.logical_and(new_mask==1, old_mask==0)

removed_roads = np.logical_and(old_mask==1, new_mask==0)

modified = np.logical_xor(old_mask, new_mask)


# COLOR VISUALIZATION


result = np.zeros((256,256,3), dtype=np.uint8)

# Green = New
result[new_roads] = [0,255,0]

# Red = Removed
result[removed_roads] = [255,0,0]

# Yellow = Modified
result[modified] = [255,255,0]


# DISPLAY


plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(old_mask, cmap="gray")
plt.title("Time T1")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(new_mask, cmap="gray")
plt.title("Time T2")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(modified, cmap="hot")
plt.title("Changed Pixels")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(result)
plt.title("Temporal Change Detection")
plt.axis("off")

plt.tight_layout()
plt.show()

print("New Road Pixels      :", np.sum(new_roads))
print("Removed Road Pixels  :", np.sum(removed_roads))
print("Modified Road Pixels :", np.sum(modified))