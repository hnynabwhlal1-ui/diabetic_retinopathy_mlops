import sys
import os
import io
import base64
import requests
import streamlit as st
from PIL import Image

# إضافة المسار الحالي لضمان استدعاء مجلد src على السحابة
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# استدعاء المحرك الداخلي للتوقع المباشر كخيار بديل تلقائي للسحابة
try:
    from src.predict import predict_and_explain
    HAS_LOCAL_PREDICT = True
except Exception as e:
    HAS_LOCAL_PREDICT = False
    IMPORT_ERROR = str(e)

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(
    page_title="AI Retinal Diagnostic Engine",
    page_icon="👁️",
    layout="wide"
)

FASTAPI_URL = "http://api:8000/predict"

# ==========================================
# 2. Advanced CSS Theme + Animated Laser Scanner
# ==========================================
def apply_custom_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #0b1329 0%, #101d36 100%);
            color: #f1f5f9;
        }
        [data-testid="stSidebar"] {
            background-color: #080e1e;
            border-right: 1px solid #1e293b;
        }
        .hero-card {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(56, 189, 248, 0.2);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            backdrop-filter: blur(4px);
            margin-bottom: 20px;
        }
        .info-box {
            background: rgba(14, 165, 233, 0.08);
            border-left: 4px solid #38bdf8;
            padding: 12px 18px;
            border-radius: 6px;
            font-size: 0.95rem;
            color: #cbd5e1;
            margin-bottom: 25px;
        }
        .scan-container {
            position: relative;
            display: inline-block;
            width: 100%;
            overflow: hidden;
            border-radius: 12px;
            border: 1px solid #38bdf8;
        }
        .scan-line {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, transparent, #00f2fe, #4facfe, transparent);
            box-shadow: 0 0 15px #00f2fe, 0 0 25px #00f2fe;
            animation: scanAnimation 2.5s infinite ease-in-out;
            z-index: 10;
        }
        @keyframes scanAnimation {
            0% { top: 0%; }
            50% { top: 95%; }
            100% { top: 0%; }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

apply_custom_theme()

# ==========================================
# 3. Hero Header & Clinical Guidance Note
# ==========================================
st.markdown(
    """
    <div class="hero-card">
        <h1 style='color: #38bdf8; margin: 0; font-size: 2.2rem; font-weight: 700;'>
            👁️ AI Retinal Screening & Diagnostic Engine
        </h1>
        <p style='color: #94a3b8; font-size: 1.05rem; margin-top: 8px;'>
            Powered by <b>FastAPI Backend Microservices</b> and <b>Explainable AI (Grad-CAM)</b>.
            Designed for real-time clinical decision support in Diabetic Retinopathy detection.
        </p>
        <div class="info-box" style="margin-top: 15px; margin-bottom: 0;">
            💡 <b>Clinical Guidance for Model Selection:</b><br>
            • <b>EfficientNetB1 (Standard RGB):</b> Optimal for high-quality retinal scans taken under standardized studio lighting.<br>
            • <b>EfficientNetB1 (Ben Graham):</b> Recommended for scans with varying contrast, non-uniform illumination, or noise artifacts, as it enhances lesion boundaries and neutralizes lighting variations.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# ==========================================
# 4. Sidebar Controls & Model Selection
# ==========================================
st.sidebar.title("🎛️ Control Panel")

model_choice = st.sidebar.selectbox(
    "Select Model Architecture",
    options=["b1_rgb", "b1_bengraham"],
    format_func=lambda x: "EfficientNetB1 (Standard RGB)" if x == "b1_rgb" else "EfficientNetB1 (Ben Graham)"
)

uploaded_files = st.sidebar.file_uploader(
    "Upload Retinal Fundus Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

st.sidebar.markdown("---")
st.sidebar.success("🟢 AI Inference Engine Ready")

# ==========================================
# 5. Diagnostic Pipeline Execution
# ==========================================
if uploaded_files:
    st.sidebar.info(f"📂 Total Scans: **{len(uploaded_files)}**")

    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        st.markdown(f"### 🖼️ Sample {idx}: `{uploaded_file.name}`")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Original Scan (Live Laser Radar)")
            st.markdown(
                """
                <div class="scan-container">
                    <div class="scan-line"></div>
                """,
                unsafe_allow_html=True
            )
            st.image(uploaded_file)
            st.markdown("</div>", unsafe_allow_html=True)

        uploaded_file.seek(0)

        with st.spinner(f"Analyzing Retinal Scan [{model_choice}]..."):
            data_processed = False
            label, confidence, gradcam_img = None, 0.0, None

            # 1. المحاولة الأولى: الاتصال بسيرفر FastAPI (إن كان يعمل)
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                params = {"model_key": model_choice}
                response = requests.post(FASTAPI_URL, files=files, params=params, timeout=3)

                if response.status_code == 200:
                    data = response.json()
                    label = data.get("prediction")
                    confidence = float(data.get("confidence", 0))
                    gradcam_b64 = data.get("gradcam_base64")
                    if gradcam_b64:
                        img_bytes = base64.b64decode(gradcam_b64)
                        gradcam_img = Image.open(io.BytesIO(img_bytes))
                    data_processed = True
            except Exception:
                data_processed = False

            # 2. المحاولة الثانية: التشغيل المباشر من predict.py عند الاستضافة السحابية
            if not data_processed and HAS_LOCAL_PREDICT:
                try:
                    uploaded_file.seek(0)
                    input_img = Image.open(uploaded_file).convert("RGB")
                    pred_res, heatmap_res = predict_and_explain(input_img, model_key=model_choice)
                    label = pred_res.get("prediction")
                    confidence = float(pred_res.get("confidence", 0))
                    gradcam_img = heatmap_res
                    data_processed = True
                except Exception as ex:
                    st.error(f"❌ Diagnostic Execution Error: {ex}")

            # 3. عرض النتائج والـ Heatmap
            if data_processed:
                with col2:
                    st.subheader("🔥 Grad-CAM Heatmap Explanation")
                    if gradcam_img is not None:
                        st.image(gradcam_img)
                    else:
                        st.warning("No Grad-CAM heatmap generated.")

                st.markdown("<br>", unsafe_allow_html=True)
                if label == "DR":
                    st.error(f"🚨 **Diagnosis:** {label} | **Confidence:** {confidence:.2f}%")
                    st.warning("⚠️ **High Risk:** Signs of Diabetic Retinopathy detected in the heatmap region.")
                else:
                    st.success(f"✅ **Diagnosis:** {label} | **Confidence:** {confidence:.2f}%")
                    st.info("ℹ️ **Low Risk:** No signs of Diabetic Retinopathy detected.")
            else:
                with col2:
                    st.error("❌ Inference Pipeline Failed")
                    if not HAS_LOCAL_PREDICT:
                        st.warning(f"⚠️ Local Predict Module failed to load: `{IMPORT_ERROR}`")
                    st.info("Ensure local models (.h5) are present in the repository or FastAPI container is reachable.")

        st.divider()
else:
    st.info("👈 Please upload retinal fundus images from the sidebar to initialize automated diagnosis.")