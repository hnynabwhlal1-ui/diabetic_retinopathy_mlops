import io
import base64
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query

# استدعاء المحرك الأساسي من مجلد src
from src.predict import run_prediction_pipeline
from src.config import MODELS_CONFIG

app = FastAPI(
    title="Diabetic Retinopathy Diagnostic Engine API",
    description="Microservice backend for DR binary classification and Grad-CAM visual explanations.",
    version="2.0"
)


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "Diabetic Retinopathy AI Diagnostic Microservice",
        "available_models": list(MODELS_CONFIG.keys())
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_key: str = Query("b1_rgb", description="Choose model key: 'b1_rgb' or 'b1_bengraham'")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")

    if model_key not in MODELS_CONFIG:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model_key. Choose from: {list(MODELS_CONFIG.keys())}"
        )

    try:
        image_bytes = await file.read()
        image_file = io.BytesIO(image_bytes)

        # 1. تشغيل المحرك الأساسي المستقل
        result = run_prediction_pipeline(image_file, model_key=model_key)

        # 2. تحويل صورة Grad-CAM إلى Base64
        gradcam_base64 = None
        if result["gradcam_image"] is not None:
            buffered = io.BytesIO()
            gradcam_pil = Image.fromarray(result["gradcam_image"])
            gradcam_pil.save(buffered, format="JPEG")
            gradcam_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "filename": file.filename,
            "model_used": model_key,
            "prediction": result["label"],
            "confidence": round(result["confidence"] * 100, 2),
            "class_index": result["class_index"],
            "raw_sigmoid": round(result["raw_value"], 4),
            "gradcam_base64": gradcam_base64,
            "status_code": 200
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing retinal image: {str(e)}")