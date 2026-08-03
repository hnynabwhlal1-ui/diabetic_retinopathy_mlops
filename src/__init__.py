import numpy as np
from PIL import Image
import tensorflow as tf
from src.config import IMG_SIZE

def preprocess_image(image_file):
    # 1. قراءة الصورة والحفاظ على ألوان RGB
    img = Image.open(image_file).convert('RGB')
    original_img = img.copy()
    
    # 2. تغيير الحجم لـ 224x224
    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized, dtype=np.float32)
    
    # 3. تطبييق preprocess_input المخصصة لـ EfficientNet تماماً كما في التدريب
    input_tensor = tf.keras.applications.efficientnet.preprocess_input(img_array)
    
    # 4. إضافة أبعاد الـ Batch -> (1, 224, 224, 3)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    return original_img, img_array, input_tensor