# Employee Burnout Prediction System (EBPS)
## Complete Presentation, Review & Deployment Guide

This guide provides a step-by-step walkthrough to present, demonstrate, and deploy the **Employee Burnout Prediction System (EBPS)** for academic or technical review.

---

## 1. Quick Project Summary & Key Accomplishments

### What is EBPS?
EBPS is an end-to-end Machine Learning / Deep Learning system designed to predict employee burnout risk over a 12-month temporal sequence using dynamic workplace metrics.

### Key Architecture Components:
1. **Deep Learning Model**: Hybrid **CNN-LSTM** architecture implemented in **TensorFlow/Keras** processing 12-month time-series feature windows (10 features/month).
2. **Feature Engineering Pipeline**: Processes raw metrics (Resource Allocation, Mental Fatigue Score, Satisfaction) and calculates temporal derivatives (Volatility Index, Satisfaction Drop, Overtime Trend).
3. **Backend API (FastAPI & Flask)**: REST endpoints for prediction, user authentication (JWT), model retraining, database querying, and data drift detection.
4. **Frontend & Dashboard (Streamlit & HTML/CSS)**: Streamlit interactive HR dashboard for risk analysis and Flask/Vanilla CSS UI.
5. **Database (PostgreSQL & SQLite)**: Persistent storage for prediction logs, user credentials, and retrained models.
6. **Docker Containerization**: Multi-container Docker environment configured with `docker-compose.yml` orchestrating PostgreSQL, FastAPI, and Streamlit.

---

## 2. Three Software Tools Explored for Review

| Tool # | Software Tool | Primary Purpose in Project | Review Demonstration Highlights |
| :--- | :--- | :--- | :--- |
| **Tool 1** | **Docker & Docker Compose** | Application Containerization & Microservice Orchestration | Multi-container isolation, `Dockerfile`, `docker-compose.yml`, health checks, volume mapping, network isolation. |
| **Tool 2** | **FastAPI + Swagger UI** | Modern Asynchronous REST API Backend & Auto-documentation | Interactive API testing page (`http://localhost:8000/docs`), Pydantic request validation, async routing. |
| **Tool 3** | **TensorFlow / Keras** | Deep Learning Model Development & Training Engine | Hybrid 1D CNN + LSTM architecture, `best_model.keras` checkpointing, model evaluation metrics & ROC curves. |

*(Alternative tools available: **Streamlit** for interactive web UI, **PostgreSQL** for relational database management).*

---

## 3. Model Training & Deep Learning Architecture

### A. Deep Learning Model Architecture (`src/model.py`)
The model uses a hybrid **1D Convolutional Neural Network + Long Short-Term Memory (CNN-LSTM)** network to capture both spatial/feature interactions and temporal progression over 12 months.

```
Input Shape: (12 Months, 10 Features)
   │
   ├── 1. Conv1D Layer (32 filters, kernel_size=3, ReLU, padding='same')
   │    └── Extracts local temporal patterns across consecutive months
   │
   ├── 2. MaxPooling1D Layer (pool_size=2)
   │    └── Downsamples sequence length while preserving peak feature signals
   │
   ├── 3. Conv1D Layer (64 filters, kernel_size=3, ReLU, padding='same')
   │    └── Higher-level spatial representation extraction
   │
   ├── 4. LSTM Layer (64 units, return_sequences=False)
   │    └── Learns long-term temporal dependencies across the 12-month period
   │
   ├── 5. Dropout Layer (rate=0.2)
   │    └── Regularization to prevent overfitting
   │
   └── 6. Dense Layer (1 unit, Sigmoid activation)
        └── Outputs burnout probability (Threshold >= 0.5 = High Risk)
```

### B. Input Features (10 Dimensions per Month)
1. `Gender_Male` (Binary)
2. `Company_Type_Service` (Binary)
3. `WFH_Setup_Available_Yes` (Binary)
4. `Designation` (Normalized 1-5 scale)
5. `Resource_Allocation` (Working hours/load scale 1-10)
6. `Mental_Fatigue_Score` (Fatigue metric scale 0-10)
7. `Satisfaction` (Employee satisfaction score 0.0 - 1.0)
8. `Volatility_Index` (Rolling standard deviation of Mental Fatigue over time)
9. `Satisfaction_Drop` (Maximum cumulative drop in satisfaction)
10. `Overtime_Trend` (Slope of Resource Allocation over past 6 months)

### C. Step-by-Step Training Demonstration Commands
To demonstrate the data preprocessing and training pipeline live in terminal:

```bash
# 1. Run full preprocessing & feature engineering pipeline
python run_pipeline.py

# 2. Train the CNN-LSTM deep learning model
python train_model.py
```

**Expected Terminal Output:**
- Pipeline loads dataset, generates 12-month sequence features, splits into Train (70%), Val (15%), Test (15%), scales continuous columns with `StandardScaler` (`scaler.pkl`), and outputs `.npy` arrays in `processed_data/`.
- Training prints epoch loss/accuracy logs, saves best model weights to `best_model.keras`, and outputs test evaluation metrics (Accuracy, F1-Score, ROC-AUC) along with loss/accuracy plots in `plots/`.

---

## 3. How to Demonstrate Docker Containers

The project uses Docker and Docker Compose to containerize PostgreSQL, FastAPI, and Streamlit into isolated microservices.

### A. Docker Services Breakdown (`docker-compose.yml`)
1. **`burnout_postgres`**: PostgreSQL 15 Alpine database container (Port `5432`).
2. **`burnout_fastapi`**: FastAPI Backend service running Uvicorn (Port `8000`).
3. **`burnout_streamlit`**: Streamlit Analytics & HR Dashboard (Port `8501`).

### B. Step-by-Step Container Demonstration Commands

#### Step 1: Build and Launch Containers
```bash
docker-compose up --build -d
```

#### Step 2: Verify Running Containers (`docker ps`)
```bash
docker ps
```
*Show the reviewer that all 3 services (`burnout_postgres`, `burnout_fastapi`, `burnout_streamlit`) are in `Up` state.*

#### Step 3: Inspect Container Logs (`docker logs`)
```bash
# Check FastAPI container logs
docker logs burnout_fastapi

# Check Streamlit container logs
docker logs burnout_streamlit

# Check Postgres container logs
docker logs burnout_postgres
```

#### Step 4: Open Live Application Endpoints in Browser
- **Streamlit HR Dashboard**: `http://localhost:8501`
- **FastAPI Interactive Swagger Docs**: `http://localhost:8000/docs`
- **FastAPI OpenAPI JSON**: `http://localhost:8000/openapi.json`
- **Flask Legacy Dashboard**: `http://localhost:5000` *(if running `python app.py`)*

#### Step 5: Clean Up / Stop Containers
```bash
docker-compose down
```

---

## 4. Everything Else Implemented in the Project

When presenting to evaluators, highlight these technical highlights:

1. **Data Drift & Monitoring (`src/drift.py`)**:
   - Calculates Kolmogorov-Smirnov (KS) test statistics between baseline training data and newly incoming inference requests to detect covariate shift over time.
2. **JWT Authentication & Security (`src/auth.py`)**:
   - Secure login, password hashing (PBKDF2/bcrypt), and token-based RBAC (Role-Based Access Control) for HR Admins vs. Regular Users.
3. **Model Retraining Trigger (`train_model.py` / API Endpoint)**:
   - Dynamic retraining endpoint allowing admins to trigger hyperparameter tuning and model updates directly from the dashboard.
4. **Publication & Evaluation Graphics (`plots/`)**:
   - Generates Training History (Loss/Accuracy curves), ROC Curves (AUC score), and Confusion Matrix heatmaps automatically.

---

## 5. Deployment Guide

### Option A: Local Docker Compose Deployment (Simplest for Demo)
Execute the following single command in terminal:
```bash
docker-compose up --build
```
Access the application at `http://localhost:8501`.

---

### Option B: Cloud Deployment on Render / Railway (Recommended for Free Cloud Hosting)

#### 1. Push Code to GitHub Repository
```bash
git add .
git commit -m "Prepare EBPS for deployment"
git push origin main
```

#### 2. Deploy Backend (FastAPI) on Render / Railway
- Create a new **Web Service** linked to your GitHub repo.
- **Environment**: Docker (`Dockerfile`).
- **Environment Variables**:
  - `DATABASE_URL`: Your PostgreSQL connection string.
- **Port**: `8000`.

#### 3. Deploy Frontend (Streamlit) on Streamlit Community Cloud
- Connect GitHub repository on [share.streamlit.io](https://share.streamlit.io).
- Set Main File Path: `streamlit_app.py`.
- Add Environment Secrets if needed.

---

### Option C: Production Deployment on AWS EC2 or VPS (Ubuntu Linux)

```bash
# 1. SSH into VPS / EC2 Instance
ssh ubuntu@your-ec2-ip

# 2. Install Docker & Docker Compose
sudo apt update && sudo apt install -y docker.io docker-compose git

# 3. Clone Repository
git clone https://github.com/krishnikumar016-coder/EBPS.git
cd EBPS

# 4. Launch Application in Background
sudo docker-compose up --build -d

# 5. Access via Server IP
# Streamlit: http://<EC2-PUBLIC-IP>:8501
# FastAPI Docs: http://<EC2-PUBLIC-IP>:8000/docs
```

---

## 6. Project Checklist for Final Review

- [x] Preprocessed data saved in `processed_data/`
- [x] Trained model saved as `best_model.keras`
- [x] High-resolution evaluation charts generated in `plots/`
- [x] Docker images defined in `Dockerfile` and `docker-compose.yml`
- [x] Streamlit dashboard tested on `http://localhost:8501`
- [x] FastAPI REST endpoints tested on `http://localhost:8000/docs`
