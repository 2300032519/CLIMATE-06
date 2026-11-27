from pathlib import Path
import numpy as np
import torch
from PIL import Image
from models.unet import UNet

def load_model(model_path):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model = UNet().to(torch.device("cpu"))
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict) and state and all(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k,v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model

def predict_road_mask(model, image, threshold=0.15):
    image = image.convert("RGB").resize((256,256), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2,0,1)).unsqueeze(0).float()
    with torch.inference_mode():
        prob = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()
    return prob, (prob > float(threshold)).astype(np.uint8)

def compare_masks(old_mask, new_mask):
    old = np.asarray(old_mask).astype(bool)
    new = np.asarray(new_mask).astype(bool)
    added = new & ~old
    removed = old & ~new
    changed = old ^ new
    rgb = np.zeros((*old.shape,3), dtype=np.uint8)
    rgb[added] = [0,220,120]
    rgb[removed] = [255,75,90]
    return {"new":added.astype(np.uint8), "removed":removed.astype(np.uint8),
            "changed":changed.astype(np.uint8), "rgb":rgb}

def overlay_mask(image, mask):
    base = np.asarray(image.convert("RGB").resize((256,256), Image.Resampling.BILINEAR)).copy()
    mask = np.asarray(mask).astype(bool)
    base[mask] = (0.55*base[mask] + 0.45*np.array([0,220,120])).astype(np.uint8)
    return base
