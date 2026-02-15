# Loan Prediction — End-to-End MLOps

A production-grade MLOps pipeline for loan approval prediction, built on Google Cloud Platform.

## Architecture

```
BigQuery / GCS (Data) → Ingestion → Validation → Preprocessing → Training → Evaluation
                                                                        ↓
                                                               Model Registry (GCS)
                                                                        ↓
                                                              FastAPI Serving API
                                                                        ↓
                                                              Monitoring (Drift)
```

## Project Structure

```
├── src/
│   ├── components/          # Pipeline steps
│   │   ├── ingestion.py     # Fetch data from GCS
│   │   ├── data_validation.py
│   │   ├── preprocess.py    # Feature engineering
│   │   ├── train.py         # Model training
│   │   └── evaluate.py      # Model evaluation
│   ├── utils/
│   │   └── config_loader.py
│   └── loan_pipeline.py     # Pipeline orchestrator
├── serve/
│   └── app.py               # FastAPI prediction API
├── monitoring/
│   └── drift_detector.py    # Data/model drift detection
├── configs/                 # YAML configuration files
├── tests/                   # Unit tests
├── Dockerfile
├── Makefile
└── pyproject.toml
```

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Authenticate with GCP
```bash
gcloud auth application-default login
```

### 3. Run Training Pipeline
```bash
python -m src.loan_pipeline
```

### 4. Start Serving API
```bash
uvicorn serve.app:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Make a Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "no_of_dependents": 2,
    "education": "Graduate",
    "self_employed": "No",
    "income_annum": 9600000,
    "loan_amount": 29900000,
    "loan_term": 12,
    "cibil_score": 778,
    "residential_assets_value": 2400000,
    "commercial_assets_value": 17600000,
    "luxury_assets_value": 22700000,
    "bank_asset_value": 8000000
  }'
```

## Docker

```bash
# Build
docker build -t loan-mlops .

# Run
docker run -p 8000:8000 loan-mlops
```

## Testing

```bash
python -m pytest tests/ -v
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/model-info` | GET | Model metadata |
| `/predict` | POST | Loan prediction |

## Tech Stack

- **ML**: scikit-learn, pandas, numpy
- **Serving**: FastAPI, Uvicorn
- **Cloud**: Google Cloud Storage, BigQuery
- **Tracking**: MLflow
- **Monitoring**: Evidently, scipy (drift detection)
- **CI/CD**: GitHub Actions
- **Containerization**: Docker