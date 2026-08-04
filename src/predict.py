import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tf_keras as keras

from src.config import MODEL_PATH, CLASS_NAMES
from src.utils import preprocess_image
from src.grad_cam import make_gradcam_heatmap, overlay_heatmap

# Load model using tf_keras
model = keras.models.load_model(MODEL_PATH, compile=False)

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