import mlflow
from src.config import MODELS_CONFIG, IMG_SIZE, CLASS_NAMES

# 1. ربط قاعدة البيانات المستهدفة
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Diabetic_Retinopathy_MLOps")

# 2. تسجيل نموذج EfficientNetB1 Standard RGB
with mlflow.start_run(run_name="EfficientNetB1_Standard_RGB"):
    mlflow.log_param("architecture", MODELS_CONFIG["b1_rgb"]["base_model_name"])
    mlflow.log_param("preprocessing", MODELS_CONFIG["b1_rgb"]["preprocessing_type"])
    mlflow.log_param("target_layer", MODELS_CONFIG["b1_rgb"]["last_conv_layer_name"])
    mlflow.log_param("image_size", str(IMG_SIZE))
    mlflow.log_param("classes", str(CLASS_NAMES))
    mlflow.log_metric("val_accuracy", 0.925)
    # تسجيل أوزان النموذج كـ Artifact
    mlflow.log_artifact(MODELS_CONFIG["b1_rgb"]["model_path"])

# 3. تسجيل نموذج EfficientNetB1 Ben Graham
with mlflow.start_run(run_name="EfficientNetB1_BenGraham"):
    mlflow.log_param("architecture", MODELS_CONFIG["b1_bengraham"]["base_model_name"])
    mlflow.log_param("preprocessing", MODELS_CONFIG["b1_bengraham"]["preprocessing_type"])
    mlflow.log_param("target_layer", MODELS_CONFIG["b1_bengraham"]["last_conv_layer_name"])
    mlflow.log_param("image_size", str(IMG_SIZE))
    mlflow.log_param("classes", str(CLASS_NAMES))
    mlflow.log_metric("val_accuracy", 0.948)
    # تسجيل أوزان النموذج كـ Artifact
    mlflow.log_artifact(MODELS_CONFIG["b1_bengraham"]["model_path"])

print("Successfully logged EfficientNetB1 models to MLflow!")