import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader

# ======================================
# IMPORT PROJECT MODULES
# ======================================

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from preprocessing.dataloader import RoadDataset
from models.unet import UNet

# ======================================
# DEVICE
# ======================================

device = torch.device("cpu")
print("Device :", device)

# ======================================
# DATASET
# ======================================

dataset = RoadDataset()

# Review-II: Use first 500 images
subset = Subset(dataset, range(1000))

train_loader = DataLoader(
    subset,
    batch_size=8,
    shuffle=True
)

print("Training Images :", len(subset))

# ======================================
# MODEL
# ======================================

model = UNet().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# ======================================
# TRAINING
# ======================================

EPOCHS = 20
loss_history = []

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.unsqueeze(1).to(device)

        outputs = model(images)

        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

# ======================================
# SAVE MODEL
# ======================================

output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "outputs"
)

os.makedirs(output_dir, exist_ok=True)

model_path = os.path.join(output_dir, "best_model.pth")

torch.save(model.state_dict(), model_path)

# ======================================
# LOSS HISTORY
# ======================================

print("\nLoss History")
for i, loss in enumerate(loss_history, start=1):
    print(f"Epoch {i}: {loss:.4f}")

print("\nTraining Completed Successfully!")
print("Model saved at:", model_path)