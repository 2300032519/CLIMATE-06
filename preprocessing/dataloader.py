import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ======================================
# DATASET PATH
# ======================================

DATASET_PATH = r"C:\CapstoneProject\Capstone\dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")

# ======================================
# CUSTOM DATASET
# ======================================

class RoadDataset(Dataset):

    def __init__(self):
        self.images = sorted(glob.glob(os.path.join(TRAIN_PATH, "*_sat.jpg")))
        self.masks = sorted(glob.glob(os.path.join(TRAIN_PATH, "*_mask.png")))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        # Read image
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read mask
        mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)

        # Resize
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        # Data Augmentation
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        if np.random.rand() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        # Normalize
        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        # HWC → CHW
        image = np.transpose(image, (2, 0, 1))

        return torch.tensor(image), torch.tensor(mask)

# =====================================
# DATALOADER
# =====================================

train_dataset = RoadDataset()

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

# =====================================
# TEST (Run only when this file is executed directly)
# =====================================

if __name__ == "__main__":

    images, masks = next(iter(train_loader))

    print("Images :", images.shape)
    print("Masks  :", masks.shape)
    print("Image Data Type :", images.dtype)
    print("Mask Data Type  :", masks.dtype)