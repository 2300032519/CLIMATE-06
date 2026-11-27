from pathlib import Path
import sys, io
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dashboard.utils import load_model, predict_road_mask, compare_masks, overlay_mask
MODEL_PATH = ROOT / "outputs" / "best_model.pth"

st.set_page_config(page_title="RoadVision AI", page_icon="🛰️", layout="wide")
css = Path(__file__).parent / "style.css"
if css.exists():
    st.markdown(f"<style>{css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

def model_or_error():
    if not MODEL_PATH.exists():
        st.error(f"Checkpoint not found: {MODEL_PATH}")
        st.info("Put best_model.pth in the project's outputs folder.")
        return None
    try:
        return get_model()
    except Exception as e:
        st.error(f"Could not load checkpoint: {e}")
        return None

with st.sidebar:
    st.markdown("## 🛰️ ROADVISION AI")
    st.caption("Road Network Intelligence")
    page = st.radio("NAVIGATION", [
        "Overview", "Road Extraction", "Temporal Change Detection",
        "Model Performance", "About Dataset"
    ])
    st.divider()
    st.caption("CAPSTONE PROJECT")

st.markdown('<div class="eyebrow">SATELLITE IMAGE ANALYTICS</div>', unsafe_allow_html=True)
st.title("Road Network Intelligence")
st.markdown('<p class="subhead">AI-powered road segmentation and temporal change analysis using satellite imagery.</p>', unsafe_allow_html=True)

if page == "Overview":
    st.subheader("Project overview")
    cols = st.columns(4)
    for col, label, value in zip(cols, ["Training images", "Architecture", "Input resolution", "Initial training"], ["6,226", "U-Net", "256 × 256", "100 images · 5 epochs"]):
        col.metric(label, value)
    st.subheader("Preliminary model evaluation")
    cols = st.columns(4)
    for col, label, value in zip(cols, ["IoU", "Dice score", "Precision", "Recall"], ["30.96%", "47.29%", "44.01%", "51.09%"]):
        col.metric(label, value)
    st.caption("Previously reported preliminary results. These are reference values, not recalculated for uploaded images.")
    st.subheader("System workflow")
    cols = st.columns(5)
    for col, n, label in zip(cols, ["01","02","03","04","05"], ["Satellite image","Preprocessing","U-Net model","Road mask","Mask comparison"]):
        col.markdown(f'<div class="stage"><span>{n}</span><b>{label}</b></div>', unsafe_allow_html=True)
    st.info("Choose Road Extraction to segment an image, or Temporal Change Detection to compare two dates.")

elif page == "Road Extraction":
    st.subheader("Road extraction")
    st.write("Upload a satellite image to generate a road probability map and binary mask.")
    file = st.file_uploader("Choose satellite image", type=["png","jpg","jpeg","tif","tiff"])
    threshold = st.slider("Road probability threshold", 0.05, 0.95, 0.15, 0.05)
    if file:
        image = Image.open(file).convert("RGB")
        model = model_or_error()
        if model is not None:
            with st.spinner("Running U-Net inference..."):
                probability, mask = predict_road_mask(model, image, threshold)
            c1,c2,c3 = st.columns(3)
            c1.image(image, caption="Original image", use_container_width=True)
            c2.image(mask * 255, caption="Predicted road mask", clamp=True, use_container_width=True)
            c3.image(overlay_mask(image, mask), caption="Road overlay", use_container_width=True)
            a,b,c = st.columns(3)
            a.metric("Input dimensions", f"{image.width} × {image.height}")
            b.metric("Predicted road pixels", f"{int(mask.sum()):,}")
            c.metric("Road pixel coverage", f"{mask.mean()*100:.2f}%")
            st.caption(f"Maximum predicted probability: {probability.max():.4f}")
            buf = io.BytesIO()
            Image.fromarray(mask * 255).save(buf, format="PNG")
            st.download_button("Download predicted mask (PNG)", buf.getvalue(), "predicted_road_mask.png", "image/png")

elif page == "Temporal Change Detection":
    st.subheader("Temporal road change detection")
    st.write("Upload two images covering the same geographic area from different dates.")
    c1,c2 = st.columns(2)
    with c1: old_file = st.file_uploader("Earlier image (T1)", type=["png","jpg","jpeg"], key="t1")
    with c2: new_file = st.file_uploader("Later image (T2)", type=["png","jpg","jpeg"], key="t2")
    threshold = st.slider("Segmentation threshold", 0.05, 0.95, 0.15, 0.05, key="th2")
    if old_file and new_file:
        old_img = Image.open(old_file).convert("RGB")
        new_img = Image.open(new_file).convert("RGB")
        st.warning("Both images should be spatially aligned and cover the same area. This prototype does not perform image registration.")
        model = model_or_error()
        if model is not None and st.button("Run temporal comparison", type="primary"):
            with st.spinner("Segmenting both images..."):
                _, old_mask = predict_road_mask(model, old_img, threshold)
                _, new_mask = predict_road_mask(model, new_img, threshold)
                changes = compare_masks(old_mask, new_mask)
            c1,c2,c3,c4 = st.columns(4)
            c1.image(old_img, caption="Satellite image T1", use_container_width=True)
            c2.image(old_mask*255, caption="Predicted mask T1", clamp=True, use_container_width=True)
            c3.image(new_img, caption="Satellite image T2", use_container_width=True)
            c4.image(new_mask*255, caption="Predicted mask T2", clamp=True, use_container_width=True)
            st.subheader("Potential changes")
            a,b,c = st.columns(3)
            a.metric("Potential new road pixels", f"{changes['new'].sum():,}")
            b.metric("Potential removed road pixels", f"{changes['removed'].sum():,}")
            c.metric("Changed pixels (XOR)", f"{changes['changed'].sum():,}")
            st.image(changes["rgb"], caption="Green = potential new roads; red = potential removed roads", use_container_width=True)
            st.caption("A binary XOR difference cannot independently establish a 'modified road' category. Changes may also result from segmentation errors or misalignment.")
            buf = io.BytesIO()
            Image.fromarray(changes["rgb"]).save(buf, format="PNG")
            st.download_button("Download change map", buf.getvalue(), "road_change_map.png", "image/png")

elif page == "Model Performance":
    st.subheader("Model performance")
    metrics = pd.DataFrame({"Metric":["IoU","Dice","Precision","Recall"],"Score (%)":[30.96,47.29,44.01,51.09]})
    fig = px.bar(metrics, x="Metric", y="Score (%)", text="Score (%)", range_y=[0,100], title="Reported preliminary segmentation metrics")
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    loss = pd.DataFrame({"Epoch":[1,2,3,4,5],"Training loss":[0.3422,0.2023,0.1947,0.1788,0.1793]})
    st.plotly_chart(px.line(loss, x="Epoch", y="Training loss", markers=True, title="Training loss over five epochs"), use_container_width=True)
    st.warning("The supplied metrics.py evaluates only the first sorted training image/mask pair at threshold 0.15. Confirm the reported values' evaluation split and protocol before treating them as final validation results.")

elif page == "About Dataset":
    st.subheader("Dataset information")
    a,b,c = st.columns(3)
    a.metric("Training images","6,226")
    b.metric("Training masks","6,226")
    c.metric("Validation images","1,243")
    st.markdown("""
    - Input: RGB satellite imagery
    - Labels: corresponding grayscale/binary road masks
    - Reported original size: 1024 × 1024 pixels
    - Model input size: 256 × 256 pixels
    - Normalization: pixel values divided by 255
    """)
    st.warning("An earlier check found zero matching validation masks. Verify validation image-mask pairing before reporting validation results. Confirm the exact dataset name and source from the downloaded dataset.")
    st.write("Technologies: Python · PyTorch · OpenCV · NumPy · Streamlit · Plotly")
