
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from src.utils.config_loader import ConfigLoader
from loggings.logger import get_logger

logger = get_logger("serve.app")

app = FastAPI(
    title="Loan Prediction Gateway API",
    description="Gateway API that forwards requests to Vertex AI Endpoint",
    version="1.0.0",
)

# Global Vertex AI Endpoint object
endpoint = None


class LoanRequest(BaseModel):
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: float
    loan_amount: float
    loan_term: int
    cibil_score: int
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    status: str = "success"


@app.on_event("startup")
def initialize_vertex_ai():
    """Initialize connection to Vertex AI Endpoint."""
    global endpoint
    try:
        config_loader = ConfigLoader()
        gcp_config = config_loader.load("gcp")
        
        project_id = gcp_config.get("gcp_login", {}).get("project_id")
        region = gcp_config.get("gcp_login", {}).get("region")
        
        # Initialize SDK
        aiplatform.init(project=project_id, location=region)
        
        # Find the deployed endpoint
        # In production, you might store the ENDPOINT_ID in env vars
        logger.info("Connecting to Vertex AI Endpoint...")
        endpoints = aiplatform.Endpoint.list(filter='display_name="loan-prediction-deployed"')
        
        if endpoints:
            endpoint = endpoints[0]
            logger.info(f"Connected to Endpoint: {endpoint.resource_name}")
        else:
            logger.warning("No deployed endpoint found with display_name='loan-prediction-deployed'")
            
    except Exception as e:
        logger.error(f"Failed to connect to Vertex AI: {e}")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "vertex_ai_connected": endpoint is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: LoanRequest):
    if endpoint is None:
        raise HTTPException(
            status_code=503, 
            detail="Vertex AI Endpoint not connected"
        )

    try:
        # Prepare instance for Vertex AI
        # The deployed container expects this format
        instance = {
            "no_of_dependents": request.no_of_dependents,
            "education": request.education,
            "self_employed": request.self_employed,
            "income_annum": request.income_annum,
            "loan_amount": request.loan_amount,
            "loan_term": request.loan_term,
            "cibil_score": request.cibil_score,
            "residential_assets_value": request.residential_assets_value,
            "commercial_assets_value": request.commercial_assets_value,
            "luxury_assets_value": request.luxury_assets_value,
            "bank_asset_value": request.bank_asset_value
        }

        logger.info(f"Forwarding request to Vertex AI Endpoint: {endpoint.resource_name}")
        
        # Send request to Vertex AI
        prediction = endpoint.predict(instances=[instance])
        
        # Parse response
        # The container returns {"prediction": "Approved", "probability": 0.95}
        # prediction.predictions is a list of results
        result = prediction.predictions[0]
        
        # Handle cases where result is a dict (custom container) or list (pre-built)
        # Custom container returns dict as we defined in original app.py
        pred_value = result if isinstance(result, str) else result.get("prediction", "Unknown")
        prob_value = 0.0
        if isinstance(result, dict):
            prob_value = result.get("probability", 0.0)
        
        return PredictionResponse(
            prediction=str(pred_value),
            probability=float(prob_value),
        )

    except Exception as e:
        logger.error(f"Vertex AI prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
