import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel 

# MLFLOW SETUP

MLFLOW_URI = "file:///C:/Users/Hrishikesh/churn-mlops-platform/mlruns"

mlflow.set_tracking_uri(MLFLOW_URI)

MODEL_NAME = "churn_champion"
MODEL_ALIAS = "champion"

model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

# FAST API
app = FastAPI(title="churn Prediction API")

# INPUT SCHEMA
class CustomerInput(BaseModel):
    gender : str
    SeniorCitizen : int
    Partner : str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# Health Check

@app.get("/")
def health():
    return {"Status":"OK", "model":MODEL_NAME}


@app.post("/predict")
def predict_churn(data:CustomerInput):
    df = pd.DataFrame([data.dict()])
    preds = model.predict(df)

    return {"churn_prabability":float(preds[0])}
