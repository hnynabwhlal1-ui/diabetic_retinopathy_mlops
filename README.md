# 👁️ Diabetic Retinopathy Diagnostic Engine: End-to-End MLOps & Explainable AI Pipeline

[![MLOps CI/CD Pipeline](https://github.com/hnynabwhlal1-ui/diabetic_retinopathy_mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/hnynabwhlal1-ui/diabetic_retinopathy_mlops/actions)
![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Keras%20%2F%20TensorFlow-red.svg)
![Backend](https://img.shields.io/badge/Backend-FastAPI-009688.svg)
![Frontend](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)
![Containerization](https://img.shields.io/badge/Container-Docker%20Compose-2496ED.svg)

An end-to-end medical AI and MLOps system engineered to classify Diabetic Retinopathy from retinal fundus imagery and provide visual diagnostic interpretability using **Grad-CAM heatmaps**. 

The application bridges deep learning research with production-grade microservice standards, featuring automated testing, Docker containerization, and continuous integration.

---

## 🔄 Evolution of the Project: From Experimentation to Production

### 📍 Phase 1: Kaggle Research & Model Suite Selection (v1.0)
During initial experimentation on Kaggle, **6 architectural variants** were trained and tracked using **MLflow**:
1. EfficientNet-B0 (Standard RGB vs. Ben Graham)
2. **EfficientNet-B1 (Standard RGB vs. Ben Graham)** *(Selected Top Performers)*
3. ResNet-50 (Standard RGB vs. Ben Graham)

Based on training convergence and validation metrics, **EfficientNet-B1** was selected as the core backbone for comparative evaluation.

---

### 🔬 Empirical Validation (External Test Set - 8 Images)
To test model generalization on real-world unstandardized clinical data (e.g., external Indian fundus datasets and web samples), both top EfficientNet-B1 models were tested on an unseen 8-image benchmark (`Test_Streamlit/`):

| Model Architecture | Image Preprocessing Method | Empirical Score | Key Clinical Observation |
| :--- | :--- | :---: | :--- |
| **EfficientNet-B1** | Standard RGB | **6 / 8 (75%)** | Struggled with lighting imbalances, background noise, and circular border artifacts on non-standard images. |
| **EfficientNet-B1** | **Ben Graham (Vein/Color Refinement)** | **7 / 8 (87.5%)** | **Outperformed RGB.** Local color subtraction and Gaussian smoothing effectively eliminated false positives on healthy eyes. |

> **Conclusion:** Ben Graham preprocessing significantly improves cross-domain generalization and visual feature clarity for diagnostic confidence.

---

### 🚀 Phase 2: Enterprise MLOps Architecture (v2.0 - Current)
The future engineering roadmap from v1.0 has been fully realized into a production-ready microservice ecosystem:

```text
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions (CI/CD Pipeline)
├── model/                         # Production Model Artifacts
├── src/                           # Core Utilities & Inference Logic
│   ├── config.py
│   ├── utils.py                   # Preprocessing & Ben Graham Filters
│   ├── grad_cam.py                # Visual Saliency Calculation
│   └── predict.py
├── tests/
│   └── test_pipeline.py           # Automated Pytest Suite
├── main.py                        # FastAPI Backend Engine
├── app.py                         # Streamlit Interactive Multi-Model UI
├── MLflow_test.py                 # MLflow Logging Scripts
├── Dockerfile                     # API Container Configuration
├── docker-compose.yml             # Multi-Container Orchestration
└── requirements.txt


🏗️ System Architecture & Microservices
The application is fully decoupled into isolated services:

Frontend Container (ui): Interactive Streamlit interface enabling clinicians to select models (RGB vs. Ben Graham), upload fundus images, inspect prediction confidence, and view Grad-CAM heatmaps.

Backend Container (api): Asynchronous FastAPI engine handling deep learning tensor transformations, inference execution, and Grad-CAM array calculations.

Docker Compose: Manages internal service networking, environment synchronization, and local deployment with zero environment drift.

🧪 Automated Testing & CI/CD Pipeline
To ensure reliability, every code push (git push origin main) triggers an automated workflow via GitHub Actions (.github/workflows/ci.yml):

Continuous Integration (CI):

Sets up a clean Linux runner (ubuntu-latest).

Installs dependencies and runs automated unit tests via pytest (test_pipeline.py).

Verifies image preprocessing tensor dimensions, model loading, and non-image error handling robustness.

Continuous Deployment Verification (CD):

Executes docker compose build to verify multi-container integration and ensure zero build breaks.

⚙️ Execution & Deployment Guide
Option 1: Docker Compose Deployment (Recommended)
Run the entire production stack with a single command:

bash
docker compose up --build

Streamlit UI: http://localhost:8501
FastAPI Docs: http://localhost:8000/docs

Option 2: Run Unit Tests Locally
Verify software stability locally before pushing:

bash
pytest tests/