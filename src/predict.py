import os
# يجب إعداد المتغيرة في أول سطر قبل استيراد أي مكتبة أخرى متعلقة بـ TensorFlow
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tf_keras as keras
from tensorflow.keras.applications import EfficientNetB0

from src.config import MODEL_PATH, CLASS_NAMES
from src.utils import preprocess_image
from src.grad_cam import make_gradcam_heatmap, overlay_heatmap

# Load model using tf_keras directly to bypass Keras 3 deserialization errors
try:
    model = keras.models.load_model(MODEL_PATH, compile=False)
except Exception:
    model = keras.models.load_model(
        MODEL_PATH, 
        compile=False, 
        custom_objects={'EfficientNetB0': EfficientNetB0}
    )

def run_prediction_pipeline(image_file):
    # Step 1: Preprocess Image
    original_img, img_array, input_tensor = preprocess_image(image_file)
    
    # Step 2: Model Inference
    predictions = model.predict(input_tensor)
    raw_val = float(predictions[0][0])  # Sigmoid Output
    
    # Debug print
    print(f"\n--- Raw Sigmoid Value: {raw_val:.4f} ---")

    # Correct Mapping Logic:
    # CLASS_NAMES = ['DR', 'No_DR'] -> Index 0 = DR, Index 1 = No_DR
    if raw_val < 0.5:
        predicted_class_idx = 0  # DR
        confidence = 1.0 - raw_val
    else:
        predicted_class_idx = 1  # No_DR
        confidence = raw_val

    predicted_label = CLASS_NAMES[predicted_class_idx]
    
    # Step 3 & 4: Grad-CAM
    heatmap = make_gradcam_heatmap(input_tensor, model)
    gradcam_result = overlay_heatmap(heatmap, np.array(original_img))
    
    return {
        "label": predicted_label,
        "confidence": float(confidence),
        "class_index": predicted_class_idx,
        "original_image": original_img,
        "gradcam_image": gradcam_result
    }

