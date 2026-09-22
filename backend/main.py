from fastapi import FastAPI, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional

from backend.schemas import (
    PredictionInput, PredictionOutput, StatsOutput,
    DatasetPageOutput, HealthOutput, PerformanceOutput
)
from backend.model_service import model_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML model and dataset
    try:
        model_service.load_resources()
    except Exception as e:
        print(f"Error initializing ModelService: {e}")
    yield
    # Shutdown logic if needed
    print("Application shutdown complete.")

app = FastAPI(
    title="Predictive Maintenance API",
    description="REST API for Industrial Machine Predictive Maintenance & Fault Classification System",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local web development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", summary="API Root Status")
def root():
    return {
        "title": "Predictive Maintenance API",
        "status": "online",
        "docs_url": "/docs"
    }

@app.get("/api/health", response_model=HealthOutput, summary="API Health Check")
def health_check():
    return {"status": "healthy"}

@app.post("/api/predict", response_model=PredictionOutput, summary="Predict Machine Failure Risk")
def predict_machine_failure(input_data: PredictionInput):
    try:
        result = model_service.predict(input_data)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/api/stats", response_model=StatsOutput, summary="Get Dataset Key Performance Statistics")
def get_dataset_stats():
    if not model_service.stats_cache:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dataset statistics not yet available."
        )
    return model_service.stats_cache

@app.get("/api/dataset", response_model=DatasetPageOutput, summary="Get Paginated Dataset Records")
def get_dataset_records(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Records per page"),
    product_type: Optional[str] = Query("All", description="Filter by Product Type (L, M, H, All)"),
    failure_status: Optional[str] = Query("All", description="Filter by Failure Status (All, Normal, Failure)")
):
    try:
        parsed_status = None
        if failure_status == "Normal" or failure_status == "0":
            parsed_status = 0
        elif failure_status == "Failure" or failure_status == "1":
            parsed_status = 1

        result = model_service.get_dataset_page(
            page=page,
            limit=limit,
            product_type=product_type,
            failure_status=parsed_status
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch dataset page: {str(e)}"
        )

@app.get("/api/charts/failure-distribution", summary="Get Machine Failure Distribution Data")
def get_failure_distribution_chart():
    data = model_service.chart_cache.get("failure_distribution")
    if not data:
        raise HTTPException(status_code=503, detail="Chart data not ready.")
    return data

@app.get("/api/charts/product-type", summary="Get Product Type Breakdown Data")
def get_product_type_chart():
    data = model_service.chart_cache.get("product_type")
    if not data:
        raise HTTPException(status_code=503, detail="Chart data not ready.")
    return data

@app.get("/api/charts/sensor-correlations", summary="Get Sensor Telemetry Correlation Samples")
def get_sensor_correlations_chart():
    data = model_service.chart_cache.get("sensor_correlations")
    if not data:
        raise HTTPException(status_code=503, detail="Chart data not ready.")
    return data

@app.get("/api/performance", response_model=PerformanceOutput, summary="Get Model Performance Evaluation Metrics")
def get_model_performance():
    try:
        metrics = model_service.get_performance_metrics()
        return metrics
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch performance metrics: {str(e)}"
        )
