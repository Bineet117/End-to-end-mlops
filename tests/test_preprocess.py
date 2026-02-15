import pytest
import pandas as pd
from src.components.preprocess import DataPreprocessor


class TestDataPreprocessor:
    def setup_method(self):
        self.preprocessor = DataPreprocessor()

    def test_preprocess_drops_loan_id(self, sample_raw_data):
        result = self.preprocessor.preprocess(sample_raw_data)
        assert "loan_id" not in result.columns

    def test_preprocess_creates_total_asset_value(self, sample_raw_data):
        result = self.preprocessor.preprocess(sample_raw_data)
        assert "total_asset_value" in result.columns

    def test_preprocess_drops_individual_assets(self, sample_raw_data):
        result = self.preprocessor.preprocess(sample_raw_data)
        dropped_cols = [
            "residential_assets_value",
            "commercial_assets_value",
            "luxury_assets_value",
            "bank_asset_value",
        ]
        for col in dropped_cols:
            assert col not in result.columns

    def test_preprocess_filters_negative_assets(self):
        df = pd.DataFrame({
            "loan_id": [1, 2],
            "no_of_dependents": [2, 0],
            "education": [" Graduate", " Not Graduate"],
            "self_employed": [" No", " Yes"],
            "income_annum": [9600000, 4100000],
            "loan_amount": [29900000, 12200000],
            "loan_term": [12, 8],
            "cibil_score": [778, 417],
            "residential_assets_value": [-100000, 2700000],
            "commercial_assets_value": [17600000, 2200000],
            "luxury_assets_value": [22700000, 8800000],
            "bank_asset_value": [8000000, 3300000],
            "loan_status": [" Approved", " Rejected"],
        })
        result = self.preprocessor.preprocess(df)
        assert len(result) == 1

    def test_mapping_encodes_loan_status(self, sample_raw_data):
        preprocessed = self.preprocessor.preprocess(sample_raw_data)
        result = self.preprocessor.mapping(preprocessed)
        assert set(result["loan_status"].unique()).issubset({0, 1})

    def test_mapping_strips_whitespace(self, sample_raw_data):
        preprocessed = self.preprocessor.preprocess(sample_raw_data)
        result = self.preprocessor.mapping(preprocessed)
        for val in result["education"].values:
            assert val == val.strip()
        for val in result["self_employed"].values:
            assert val == val.strip()
