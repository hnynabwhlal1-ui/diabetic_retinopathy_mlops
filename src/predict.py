import os
import json
import zipfile
import tempfile
from functools import lru_cache
import numpy as np
import tensorflow as tf

from src.config import CLASS_NAMES, MODELS_CONFIG
from src.utils import preprocess_image
from src.grad_cam import make_gradcam_heatmap, overlay_heatmap


def clean_config(config_obj):
    """Recursively remove quantization_config keys causing Keras compatibility issues."""
    if isinstance(config_obj, dict):
        config_obj.pop("quantization_config", None)
        for key, value in config_obj.items():
            clean_config(value)
    elif isinstance(config_obj, list):
        for item in config_obj:
            clean_config(item)


@lru_cache(maxsize=2)
def load_robust_model(model_path):
    """Load model with automated configuration sanitization and LRU memory caching."""
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"Standard loading failed for {model_path}. Sanitizing config.json...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(model_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            config_file_path = os.path.join(temp_dir, "config.json")
            if os.path.exists(config_file_path):
                with open(config_file_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                
                clean_config(config_data)
                
                with open(config_file_path, "w", encoding="utf-8") as f:
                    json.dump(config_data, f)
            
            sanitized_model_path = os.path.join(temp_dir, "sanitized_model.keras")
            with zipfile.ZipFile(sanitized_model_path, 'w', zipfile.ZIP_DEFLATED) as zip_out:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        if file == "sanitized_model.keras":
                            continue
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, temp_dir)
                        zip_out.write(full_path, rel_path)
            
            return tf.keras.models.load_model(sanitized_model_path, compile=False)


def run_prediction_pipeline(image_file, model_key="b1_rgb"):
    """
    Core Inference & Grad-CAM Pipeline.
    Supports dynamic switching between 'b1_rgb' and 'b1_bengraham'.
    """
    # 1. جلب إعدادات النموذج المختار (RGB أو BenGraham)
    config = MODELS_CONFIG.get(model_key, MODELS_CONFIG["b1_rgb"])
    model_path = config["model_path"]
    method = config["preprocessing_type"]
    base_model_name = config["base_model_name"]
    last_conv_layer_name = config["last_conv_layer_name"]

    # 2. تحميل النموذج المطلوب
    model = load_robust_model(model_path)

    # 3. معالجة الصورة بنفس الطريقة الخاصة بالنموذج (سواء BenGraham أو RGB عادي)
    original_img, img_array, input_tensor = preprocess_image(image_file, method=method)
    
    # 4. التنبؤ حساب قيم Sigmoid
    predictions = model.predict(input_tensor)
    raw_val = float(predictions[0][0])

    # 5. التصنيف بناءً على شروط النوت بوك (Index 0 = DR, Index 1 = No_DR)
    if raw_val < 0.5:
        predicted_class_idx = 0  # DR
        confidence = 1.0 - raw_val
    else:
        predicted_class_idx = 1  # No_DR
        confidence = raw_val

    predicted_label = CLASS_NAMES[predicted_class_idx]
    
    # 6. توليد Grad-CAM بناءً على الطبقة المحددة للنموذج المختار
    try:
        heatmap = make_gradcam_heatmap(
            input_tensor, 
            model, 
            last_conv_layer_name=last_conv_layer_name,
            base_model_name=base_model_name
        )
        gradcam_result = overlay_heatmap(heatmap, np.array(original_img))
    except Exception as e:
        print(f"Grad-CAM generation warning: {e}")
        gradcam_result = original_img
    
    return {
        "label": predicted_label,
        "confidence": float(confidence),
        "class_index": predicted_class_idx,
        "original_image": original_img,
        "gradcam_image": gradcam_result,
        "raw_value": raw_val
    }