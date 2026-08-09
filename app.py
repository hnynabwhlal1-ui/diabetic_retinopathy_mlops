import streamlit as st
from PIL import Image
import numpy as np
from src.predict import run_prediction_pipeline
from src.config import CLASS_NAMES

# ==========================================
# 1. Page Configuration & Setup
# ==========================================
# Configures the browser tab title, icon, and wide layout mode for dual-column view.
st.set_page_config(
    page_title="Diabetic Retinopathy Diagnostic System",
    page_icon="👁️",
    layout="wide"
)

# ==========================================
# 2. Custom CSS Styling (Dark Theme)
# ==========================================
# Applies custom CSS to construct a modern, dark-themed medical diagnostic UI.
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

# ==========================================
# 3. Header and Title Section
# ==========================================
st.title("👁️ Diabetic Retinopathy Detection & Explainable AI (Grad-CAM)")
st.markdown("""
This application uses a Deep Learning model (**EfficientNetB0**) combined with **Grad-CAM**
to classify retinal images and highlight the specific areas that influenced the decision.
""")

st.divider()

# ==========================================
# 4. Multi-File Uploader Component
# ==========================================
# Set accept_multiple_files=True to allow selecting and uploading up to 8+ images at once.
uploaded_files = st.sidebar.file_uploader(
    "Upload Retinal Fundus Images (Select Multiple)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# ==========================================
# 5. Iterative Batch Processing Pipeline
# ==========================================
# If files are uploaded, loop through each image individually and display predictions.
if uploaded_files:
    st.sidebar.success(f"📂 Total Images Loaded: **{len(uploaded_files)}**")
    
    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        st.markdown(f"### 🖼️ Sample {idx}: `{uploaded_file.name}`")
        
        # Create two columns to display original image and Grad-CAM explanation side-by-side
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Original Retinal Image")
            st.image(uploaded_file, use_container_width=True)

        # Reset the byte stream pointer before passing the file object to the inference function
        uploaded_file.seek(0)

        with st.spinner(f"Analyzing Image {idx}/{len(uploaded_files)} and generating Grad-CAM..."):
            results = run_prediction_pipeline(uploaded_file)

        # Graceful error handling for missing keys or pipeline failure
        if "error" in results:
            st.error(f"❌ Error in {uploaded_file.name}: {results['error']}")
            st.divider()
            continue

        with col2:
            st.subheader("🔥 Grad-CAM Heatmap Explanation")
            st.image(results["gradcam_image"], use_container_width=True)

        # Display Diagnostic Results & Confidence Score
        label = results["label"]
        confidence = results["confidence"] * 100

        if label == "DR":
            st.error(f"**Diagnosis:** {label} | **Confidence:** {confidence:.2f}%")
            st.warning("⚠️ High Risk: Signs of Diabetic Retinopathy detected in the heatmap region.")
        else:
            st.success(f"**Diagnosis:** {label} | **Confidence:** {confidence:.2f}%")
            st.info("✅ Low Risk: No signs of Diabetic Retinopathy detected.")

        st.divider()
