import os
import numpy as np
import tensorflow as tf

from src.config import MODEL_PATH, CLASS_NAMES
from src.utils import preprocess_image
from src.grad_cam import make_gradcam_heatmap, overlay_heatmap

def load_robust_model(model_path):
    try:
        # المحاولة الأولى: التحميل المباشر
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        # إذا فشل التفريغ البنائي، نبني المعمارية ونحمل الأوزان فقط
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None, input_shape=(224, 224, 3)
        )
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        built_model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
        
        # تحميل الأوزان فقط وتجاهل هيكل الطبقات المعطوب
        built_model.load_weights(model_path)
        return built_model

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