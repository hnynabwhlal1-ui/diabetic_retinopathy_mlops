import os
import json
import zipfile
import tempfile
import numpy as np
import tensorflow as tf

from src.config import MODEL_PATH, CLASS_NAMES
from src.utils import preprocess_image
from src.grad_cam import make_gradcam_heatmap, overlay_heatmap

def clean_config(config_obj):
    """Recursively remove quantization_config keys that cause local Keras compatibility issues."""
    if isinstance(config_obj, dict):
        config_obj.pop("quantization_config", None)
        for key, value in config_obj.items():
            clean_config(value)
    elif isinstance(config_obj, list):
        for item in config_obj:
            clean_config(item)

def load_robust_model(model_path):
    try:
        # Attempt standard loading first
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"Standard loading failed due to config mismatch. Sanitizing config.json...")
        
        # Create a temporary directory to extract zip and modify config
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(model_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            config_file_path = os.path.join(temp_dir, "config.json")
            if os.path.exists(config_file_path):
                with open(config_file_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                
                # Sanitize configuration object
                clean_config(config_data)
                
                with open(config_file_path, "w", encoding="utf-8") as f:
                    json.dump(config_data, f)
            
            # Re-compress into a clean temporary model archive
            sanitized_model_path = os.path.join(temp_dir, "sanitized_model.keras")
            with zipfile.ZipFile(sanitized_model_path, 'w', zipfile.ZIP_DEFLATED) as zip_out:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        if file == "sanitized_model.keras":
                            continue
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, temp_dir)
                        zip_out.write(full_path, rel_path)
            
            # Load sanitized model
            return tf.keras.models.load_model(sanitized_model_path, compile=False)

# Load model with automatic sanitization handling
model = load_robust_model(MODEL_PATH)

def run_prediction_pipeline(image_file):
    original_img, img_array, input_tensor = preprocess_image(image_file)
    
    predictions = model.predict(input_tensor)
    raw_val = float(predictions[0][0])
    
    print(f"\n--- Raw Sigmoid Value: {raw_val:.4f} ---")

    if raw_val < 0.5:
        predicted_class_idx = 0
        confidence = 1.0 - raw_val
    else:
        predicted_class_idx = 1
        confidence = raw_val

    predicted_label = CLASS_NAMES[predicted_class_idx]
    
    heatmap = make_gradcam_heatmap(input_tensor, model)
    gradcam_result = overlay_heatmap(heatmap, np.array(original_img))
    
    return {
        "label": predicted_label,
        "confidence": float(confidence),
        "class_index": predicted_class_idx,
        "original_image": original_img,
        "gradcam_image": gradcam_result
    }