import streamlit as st
from PIL import Image
import numpy as np
from src.predict import run_prediction_pipeline
from src.config import CLASS_NAMES

# 1. Page Configuration
st.set_page_config(
    page_title="Diabetic Retinopathy Diagnostic System",
    page_icon="👁️",
    layout="wide"
)

# --- CUSTOM CSS FOR BEAUTIFUL DARK MEDICAL THEME ---
def apply_custom_theme():
    st.markdown(
        """
        <style>
        /* Modern Dark Gradient Background */
        .stApp {
            background: linear-gradient(135deg, #0d131a 0%, #17212b 50%, #1e2d3d 100%);
            color: #e2e8f0;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #111822;
            border-right: 1px solid #1e293b;
        }

        /* Card-like borders for images and results */
        div[data-testid="stBlock"] {
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

apply_custom_theme()
# --------------------------------------------------

# 2. Header and Title
st.title("👁️ Diabetic Retinopathy Detection & Explainable AI (Grad-CAM)")
st.markdown("""
This application uses a Deep Learning model (**EfficientNetB0**) combined with **Grad-CAM**
to classify retinal images and highlight the specific areas that influenced the decision.
""")

st.divider()

# 3. File Uploader Component
uploaded_file = st.sidebar.file_uploader(
    "Upload a Retinal Fundus Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Create two columns to display original image and Grad-CAM explanation side-by-side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original Retinal Image")
        st.image(uploaded_file, use_container_width=True)

    # IMPORTANT: reset the stream position before reading the file again,
    # since st.image() above already consumed it.
    uploaded_file.seek(0)

    with st.spinner("Analyzing image and generating Grad-CAM explanation..."):
        results = run_prediction_pipeline(uploaded_file)

    # Stop here and show a clear error if the pipeline failed,
    # instead of crashing on a missing key.
    if "error" in results:
        st.error(f"❌ {results['error']}")
        st.stop()

    with col2:
        st.subheader("🔥 Grad-CAM Heatmap Explanation")
        st.image(results["gradcam_image"], use_container_width=True)

    st.divider()

    # 4. Display Diagnostic Results
    st.subheader("📊 Diagnostic Summary")

    label = results["label"]
    confidence = results["confidence"] * 100

    # Compare by label name (not by index) so this stays correct
    # even if CLASS_NAMES order in config.py ever changes.
    if label == "DR":
        st.error(f"**Diagnosis:** {label} | **Confidence:** {confidence:.2f}%")
        st.warning("⚠️ High Risk: Signs of Diabetic Retinopathy detected in the heatmap region.")
    else:
        st.success(f"**Diagnosis:** {label} | **Confidence:** {confidence:.2f}%")
        st.info("✅ Low Risk: No signs of Diabetic Retinopathy detected.")