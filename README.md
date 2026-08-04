# 👁️ Diabetic Retinopathy Detection: End-to-End MLOps & Explainable AI Pipeline

[![Basic CI Pipeline](https://github.com/hnynabwhlal1-ui/diabetic_retinopathy_mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/hnynabwhlal1-ui/diabetic_retinopathy_mlops/actions)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![Keras Framework](https://img.shields.io/badge/Framework-Keras%20%2F%20TensorFlow-red.svg)
![Streamlit App](https://img.shields.io/badge/Deployment-Streamlit-FF4B4B.svg)

An end-to-end Computer Vision and MLOps engineering system designed to detect and grade Diabetic Retinopathy from retinal fundus imagery. The system bridges medical interpretability (Explainable AI) with production-grade software standards, featuring automated testing and Continuous Integration (CI).

---

## 🛠️ System Architecture & Directory Structure

The project transitions standard deep learning scripts into a production-ready, modular codebase (`src/`) integrated with continuous quality control.

```text
├── .github/
│   └── workflows/
│       └── ci.yml              # Cloud Automation & CI Pipeline (GitHub Actions)
├── model/
│   └── trained_model.keras     # Deep Learning Model Artifact
├── src/                        # Core Modular Codebase
│   ├── config.py               # Constants, Paths, and Target Class Specifications
│   ├── utils.py                # Preprocessing Protocols & Image Array Conversions
│   ├── grad_cam.py             # Gradient Activation Heatmap Calculations
│   └── predict.py              # End-to-End Prediction Pipeline Execution
├── Test_Streamlit/             # Local Empirical Validation Dataset (8 Images)
├── tests/
│   └── test_pipeline.py        # Automated Test Suites (Pytest)
├── app.py                      # Interactive Web User Interface (Streamlit)
└── requirements.txt            # Dependency Management & Environment Lock


# 🩺 Explainable AI (Grad-CAM) & Clinical Value

In clinical decision-support systems, prediction confidence is bound to model explainability. A raw classification score is insufficient for medical validation.

* Diagnostic Verification: Integrated Gradient-weighted Class Activation Mapping (Grad-CAM) computes the visual saliency map directly from the final convolutional layers.

* Streamlit Integration: Upon image processing, the UI renders both the predicted classification stage and the overlaid heatmap. This highlights lesion patterns (such as microaneurysms, exudates, and hemorrhages), giving clinicians immediate visual evidence to verify AI outputs against diagnostic standards.

# 🔬 Empirical Validation (Test_Streamlit) & Ben Graham MotivationTo 
evaluate domain generalization, the Streamlit interface was tested using a curated benchmark folder Test_Streamlit containing 8 test images split into two categories:

## 📊 Validation Results & Insights:
1. Diabetic Retinopathy (DR) — 4 Images:
* Result: 4/4 Correctly Classified (100% Accuracy).
* Observation: Both the 2 standard benchmark dataset images and the 2 random external web images were accurately identified due to prominent, high-contrast lesion features.

2. No Retinopathy (No DR) — 4 Images:
* Result: 2 Correct / 2 Incorrect (False Positives).
* Observation: The 2 images sourced from standard datasets were correctly predicted as No DR. However, the 2 random external web images resulted in False Positives (misclassified as DR).

# 💡 Justification for Future Integration (Ben Graham Method):
Unstandardized web images suffer from background noise, camera artifacts, and lighting imbalances that the model mistakes for lesions. Integrating Ben Graham Preprocessing (local color subtraction + Gaussian smoothing) in future iterations will:

* Remove uninformative circular background margins.
* Equalize non-uniform illumination across different camera sources to eliminate False Positives on healthy eyes.

# 🧪 Quality Assurance & Continuous Integration (CI)
To eliminate pipeline fragility, avoid runtime breaking changes, and guarantee cross-environment stability, the project implements rigorous automated software testing.

1. Automated Unit & Robustness Testing (Pytest)
* test_preprocess_image_shape: Verifies that uploaded binary image streams are correctly parsed, resized, normalized, and transformed into appropriate model tensors without silently dropping bytes.

* test_model_prediction_pipeline: Ensures the trained .keras model artifact loads into memory seamlessly, processes the feature vector, and outputs expected inference probability arrays.

* test_invalid_image_input: Evaluates system resilience (Robustness Testing) against corrupted inputs or non-image file uploads, enforcing safe exception handling rather than unhandled system crashes.

2. Cloud Automation (GitHub Actions)
Every code push to the main branch triggers a headless Linux environment (ubuntu-latest) via .github/workflows/ci.yml. The runner installs exact runtime dependencies and executes pytest automatically, preventing broken builds from reaching deployment.

# 🔮 Future Engineering Roadmap
1. Ben Graham Preprocessing: Implementing Gaussian blurring and local color subtraction algorithms as proven by Test_Streamlit empirical findings.

2. Containerization (Docker): Packaging the code, model binaries, and runtime environment into Docker images to guarantee uniform execution across different hosting infrastructure.

3. ML Experiment Tracking (MLflow): Tracking dataset lineage, model versioning, hyperparameters, and evaluation metrics during prospective model retraining cycles.


## ⚙️ Execution Guide

### 1. Installation
Clone the repository and set up runtime dependencies:
```bash
git clone https://github.com/hnynabwhlal1-ui/diabetic_retinopathy_mlops.git
cd diabetic_retinopathy_mlops
pip install -r requirements.txt

2. Run Automated Testing Framework
Execute local test suites via Pytest:

pytest

3. Launch Application
Start the Streamlit web application:


streamlit run app.py