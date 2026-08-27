import os

# General Image Dimensions for EfficientNetB1
IMG_SIZE = (224, 224)
IMG_SHAPE = (224, 224, 3)

# Diagnostic Class Names: Index 0 -> DR, Index 1 -> No_DR
CLASS_NAMES = ['DR', 'No_DR']

# Base Directory Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Multi-Model Configurations (RGB + BenGraham)
MODELS_CONFIG = {
    "b1_rgb": {
        "model_path": os.path.join(BASE_DIR, "model", "eye_disease_diagnostic_model.keras"),
        "base_model_name": "EfficientNetB1",
        "last_conv_layer_name": "top_activation",
        "preprocessing_type": "rgb"
    },
    "b1_bengraham": {
        "model_path": os.path.join(BASE_DIR, "model", "best_Exp4_EfficientNetB1_BenGraham.keras"),
        "base_model_name": "EfficientNetB1",
        "last_conv_layer_name": "top_activation",
        "preprocessing_type": "bengraham"
    }
}

# Default Active Model Setup
DEFAULT_MODEL_KEY = "b1_rgb"

MODEL_PATH = MODELS_CONFIG[DEFAULT_MODEL_KEY]["model_path"]
BASE_MODEL_NAME = MODELS_CONFIG[DEFAULT_MODEL_KEY]["base_model_name"]
LAST_CONV_LAYER_NAME = MODELS_CONFIG[DEFAULT_MODEL_KEY]["last_conv_layer_name"]