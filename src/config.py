import os

# The dimensions of the image EfficientNetB0 expects
IMG_SIZE = (224,224)
IMG_SHAPE = (224,224,3)

# Index 0 -> DR , Index 1 -> No_DR
CLASS_NAMES = ['DR', 'No_DR']

#The path of Saving and retrieving the form 
MODEL_PATH = os.path.join("model","best_diabetic_retinopathy_model.keras")

#The name of the last Grad_CAM layer
BASE_MODEL_NAME = "efficientnetb0"
LAST_CONV_LAYER_NAME = "top_conv"