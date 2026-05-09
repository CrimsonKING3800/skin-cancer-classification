"""
app.py - Streamlit demo for Skin Cancer Classification.

Usage: streamlit run demo/app.py
"""
import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="Skin Cancer Classifier", page_icon="🩺", layout="centered")

st.title("🩺 Skin Cancer Classification")
st.markdown("Upload a dermoscopic image to get a malignancy prediction.")
st.markdown("---")

uploaded = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing..."):
        # Placeholder - replace with actual model inference
        img_arr = np.array(img.resize((224, 224))) / 255.0
        # probability = model.predict(img_arr[np.newaxis, ...])[0][0]
        probability = np.random.uniform(0.1, 0.9)  # demo placeholder

    st.markdown("### Results")
    if probability >= 0.5:
        st.error(f"🔴 **MALIGNANT** (Probability: {probability:.2%})")
    else:
        st.success(f"🟢 **BENIGN** (Probability of malignancy: {probability:.2%})")

    st.progress(probability)
    st.caption("⚠️ This is a demo. Load a trained model for real predictions.")
