import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_raw_data():
    """Create sample raw loan data for testing."""
    return pd.DataFrame({
        "loan_id": [1, 2, 3, 4, 5],
        "no_of_dependents": [2, 0, 3, 3, 5],
        "education": [" Graduate", " Not Graduate", " Graduate", " Graduate", " Not Graduate"],
        "self_employed": [" No", " Yes", " No", " No", " Yes"],
        "income_annum": [9600000, 4100000, 9100000, 8200000, 9800000],
        "loan_amount": [29900000, 12200000, 29700000, 30700000, 24200000],
        "loan_term": [12, 8, 20, 8, 20],
        "cibil_score": [778, 417, 506, 467, 382],
        "residential_assets_value": [2400000, 2700000, 7100000, 18200000, 12400000],
        "commercial_assets_value": [17600000, 2200000, 4500000, 3300000, 8200000],
        "luxury_assets_value": [22700000, 8800000, 33300000, 23300000, 29400000],
        "bank_asset_value": [8000000, 3300000, 12800000, 7900000, 5000000],
        "loan_status": [" Approved", " Rejected", " Rejected", " Rejected", " Rejected"],
    })


@pytest.fixture
def sample_preprocessed_data():
    """Create sample preprocessed data for testing."""
    return pd.DataFrame({
        "no_of_dependents": [2, 0, 3, 3, 5],
        "education": ["Graduate", "Not Graduate", "Graduate", "Graduate", "Not Graduate"],
        "self_employed": ["No", "Yes", "No", "No", "Yes"],
        "income_annum": [9600000, 4100000, 9100000, 8200000, 9800000],
        "loan_amount": [29900000, 12200000, 29700000, 30700000, 24200000],
        "loan_term": [12, 8, 20, 8, 20],
        "cibil_score": [778, 417, 506, 467, 382],
        "total_asset_value": [50700000, 17000000, 57700000, 52700000, 55000000],
        "loan_status": [1, 0, 0, 0, 0],
    })
