import tensorflow as tf
import numpy as np
from PIL import Image
from src.config import IMG_SIZE

def preprocess_image(image_file):
    """
    قراءة صورة العين وتعديل مقاسها وتحويلها لـ Tensor جاهز للتوقع
    """
    # فتح الصورة باستخدام PIL
    img = Image.open(image_file).convert('RGB')
    
    # تغيير المقاس إلى 224x224
    img = img.resize(IMG_SIZE)
    
    # تحويل الصورة إلى مصفوفة NumPy
    img_array = np.array(img, dtype=np.float32)
    
    # إضافة بعد الـ Batch (1, 224, 224, 3)
    input_tensor = np.expand_dims(img_array, axis=0)
    
    return img, img_array, input_tensor