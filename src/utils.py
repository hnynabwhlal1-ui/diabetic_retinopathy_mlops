import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from src.config import IMG_SIZE

def clean_standard_preprocess(image_input):
    """
    Standard Clean Preprocessing (RGB + Resizing)
    Used primarily for standard high-quality images.
    """
    if isinstance(image_input, bytes):
        file_bytes = np.asarray(bytearray(image_input), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        image = np.array(image_input.convert('RGB'))
    else:
        image = image_input.copy()

    image_resized = cv2.resize(image, IMG_SIZE)
    return image_resized

def ben_graham_preprocess(image_input, sigmaX=10):
    """
    Ben Graham Preprocessing (Color & Contrast Enhancement)
    Enhances vascular contrast for low-light or noisy fundus images.
    """
    image_rgb = clean_standard_preprocess(image_input)
    # Crop outer black boundaries & apply Gaussian Blur subtraction
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX)
    enhanced = cv2.addWeighted(image_bgr, 4, blurred, -4, 128)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    
    return enhanced_rgb

def preprocess_image(image_file, method="rgb"):
    """
    Unified Image Preprocessing Entry Point
    Accepts method='rgb' or method='bengraham'
    """
    if hasattr(image_file, 'read'):
        image_bytes = image_file.read()
    else:
        image_bytes = image_file

    # Route based on selected pipeline strategy
    if method == "bengraham":
        processed_np = ben_graham_preprocess(image_bytes)
    else:
        processed_np = clean_standard_preprocess(image_bytes)

    # Convert array back to PIL for Streamlit UI display
    pil_img = Image.fromarray(processed_np)

    # Convert to Keras-compatible 4D Tensor batch (1, 224, 224, 3)
    img_array = processed_np.astype(np.float32)
    input_tensor = np.expand_dims(img_array, axis=0)
    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)

    return pil_img, img_array, input_tensor