from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class PredictionInput(BaseModel):
    type: str = Field(default="M", description="Product Quality Variant (L, M, H)")
    air_temperature: float = Field(default=298.1, description="Air temperature in Kelvin")
    process_temperature: float = Field(default=308.6, description="Process temperature in Kelvin")
    rotational_speed: int = Field(default=1500, description="Rotational speed in RPM")
    torque: float = Field(default=40.0, description="Torque in Nm")
    tool_wear: int = Field(default=0, description="Tool wear duration in minutes")
    twf: int = Field(default=0, description="Tool Wear Failure indicator (0 or 1)")
    hdf: int = Field(default=0, description="Heat Dissipation Failure indicator (0 or 1)")
    pwf: int = Field(default=0, description="Power Failure indicator (0 or 1)")
    osf: int = Field(default=0, description="Overstrain Failure indicator (0 or 1)")
    rnf: int = Field(default=0, description="Random Failure indicator (0 or 1)")

class PredictionOutput(BaseModel):
    prediction: int
    status: str
    risk_level: str
    failure_probability: float
    normal_probability: float
    feature_importances: Dict[str, float]

class StatsOutput(BaseModel):
    total_machines: int
    total_failures: int
    normal_machines: int
    failure_rate: float
    total_columns: int
    missing_values: int

class DatasetRecord(BaseModel):
    UDI: int
    Product_ID: str = Field(alias="Product ID")
    Type: str
    Air_temperature_K: float = Field(alias="Air temperature [K]")
    Process_temperature_K: float = Field(alias="Process temperature [K]")
    Rotational_speed_rpm: int = Field(alias="Rotational speed [rpm]")
    Torque_Nm: float = Field(alias="Torque [Nm]")
    Tool_wear_min: int = Field(alias="Tool wear [min]")
    Machine_failure: int = Field(alias="Machine failure")
    TWF: int
    HDF: int
    PWF: int
    OSF: int
    RNF: int

    class Config:
        populate_by_name = True

class DatasetPageOutput(BaseModel):
    page: int
    limit: int
    total_records: int
    total_pages: int
    records: List[Dict[str, Any]]

class HealthOutput(BaseModel):
    status: str

class PerformanceOutput(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: List[List[int]]
    feature_importances: List[Dict[str, Any]]
