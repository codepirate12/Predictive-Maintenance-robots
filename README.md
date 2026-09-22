<<<<<<< HEAD
# Predictive Maintenance and Fault Classification System ⚙️

A modern, full-stack web application for predicting industrial machine failure risks based on telemetry and sensor data from the **AI4I 2020 Predictive Maintenance Dataset**.

This project replaces the old Streamlit UI with a **FastAPI** Python REST backend and a **HTML5 / CSS3 / Vanilla JavaScript** dashboard frontend powered by **Chart.js**.

---

## 📌 Project Features

1. **Interactive Dashboard**: KPI Metrics (`Total Machines`, `Recorded Failures`, `Normal Machines`, `Failure Rate`) and 4 interactive Chart.js graphs (Failure Distribution, Failures by Product Type, Torque vs Rotational Speed Scatter, Tool Wear Distribution).
2. **Real-Time Failure Risk Classifier**: Sensor input form with quick-fill presets (`Load Normal Machine Preset`, `Load High Risk Preset`), animated probability progress bars, risk category badges (`Low`, `Medium`, `High`), parameter summary tables, and disclaimers.
3. **Dataset Explorer**: Paginated table displaying records from the 10,000 industrial machine telemetry dataset with Product Type and Machine Failure filters.
4. **Model Performance Analytics**: Verified classification metrics (`Accuracy=87%`, `Precision=0.83`, `Recall=0.72`, `F1-Score=0.77`), Confusion Matrix grid, Scikit-Learn Classification Report table, and Random Forest Feature Importances.
5. **About Project & ML Pipeline**: System architecture overview, AI4I dataset schema, tech stack badges, and an interactive 5-step machine learning workflow diagram.

---

## 🏗️ Project Structure

```text
project/
├── backend/
│   ├── main.py                # FastAPI REST API endpoints & CORS middleware
│   ├── model_service.py       # ML Model loading, preprocessing & dataset query service
│   ├── schemas.py             # Pydantic data validation schemas
│   └── requirements.txt       # Backend dependencies
│
├── frontend/
│   ├── index.html             # Single-Page Dashboard HTML layout
│   ├── style.css              # Custom industrial dark theme CSS
│   ├── script.js              # Vanilla JavaScript router, Chart.js integrations & API client
│   └── assets/                # Visual icons and assets
│
├── models/
│   └── predictive_maintenance_model.pkl  # Pre-trained Random Forest ML model
│
├── data/
│   └── ai4i2020.csv           # 10,000 row AI4I dataset
│
├── README.md                  # Project documentation
└── .gitignore                 # Standard Python & web gitignore
```

---

## 🤖 Machine Learning Model & Preprocessing

- **Algorithm**: `RandomForestClassifier` (`predictive_maintenance_model.pkl`)
- **Expected Features (12)**:
  `['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'Type_L', 'Type_M']`
- **Categorical Encoding**: Product Type `L` maps to `Type_L=1`, Product Type `M` maps to `Type_M=1`, Product Type `H` serves as baseline (`Type_L=0, Type_M=0`).

---

## 🚀 How to Run the Application

### 1. Install Backend Dependencies

```bash
pip install -r backend/requirements.txt
```

### 2. Start the FastAPI Backend Server

Run using `python -m uvicorn` from the root directory (this bypasses Windows PowerShell PATH issues):

```powershell
python -m uvicorn backend.main:app --reload --port 8000
```

> **Note on PowerShell `uvicorn` Command**:
> Typing `uvicorn` directly in Windows PowerShell can cause `CommandNotFoundException` if `%APPDATA%\Python\Python311\Scripts` is not added to system environment variables. Running `python -m uvicorn backend.main:app --reload --port 8000` bypasses PATH errors.
> Alternatively, double-click `run_backend.bat`.

- API Base URL: `http://localhost:8000`
- Interactive Swagger API Documentation: `http://localhost:8000/docs`

### 3. Launch the Frontend Dashboard

Run the Python HTTP server:

```powershell
python -m http.server 5500 --directory frontend
```
> Or double-click `run_frontend.bat`.

Open your browser at:
`http://localhost:5500`

---

## 🔌 API Endpoints Summary

- `GET /` — API root status.
- `GET /api/health` — System health check (`{"status": "healthy"}`).
- `POST /api/predict` — Submit machine parameters and receive binary prediction, failure probability, and risk level.
- `GET /api/stats` — Retrieve dataset KPI counts and statistics.
- `GET /api/dataset` — Retrieve paginated dataset records (`?page=1&limit=25&product_type=All&failure_status=All`).
- `GET /api/charts/failure-distribution` — Aggregated failure distribution data for doughnut chart.
- `GET /api/charts/product-type` — Failure breakdown counts by Product Type.
- `GET /api/charts/sensor-correlations` — Sampled telemetry correlation points for scatter and histogram plots.
- `GET /api/performance` — Model evaluation metrics (`Accuracy`, `Precision`, `Recall`, `F1`, Confusion Matrix, Feature Importances).
=======
# Predictive-Maintenance-robots
>>>>>>> 58e604184ef56708c81f09ff10b53e184b8503d1
