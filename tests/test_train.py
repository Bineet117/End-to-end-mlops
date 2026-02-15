import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.components.train import ModelTrainer


class TestModelTrainer:
    def setup_method(self):
        self.trainer = ModelTrainer()

    def test_encode_features(self, sample_preprocessed_data):
        df = sample_preprocessed_data.copy()
        # Remove loan_status mapping since encode_features expects string categoricals
        df["loan_status"] = [1, 0, 0, 0, 0]
        result = self.trainer.encode_features(df)
        assert result["education"].dtype in [np.int32, np.int64, int]
        assert result["self_employed"].dtype in [np.int32, np.int64, int]

    def test_encode_features_stores_encoders(self, sample_preprocessed_data):
        df = sample_preprocessed_data.copy()
        self.trainer.encode_features(df)
        assert "education" in self.trainer.label_encoders
        assert "self_employed" in self.trainer.label_encoders

    def test_scale_features(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        X_scaled = self.trainer.scale_features(X)
        # Scaled data should have mean ~0 and std ~1
        assert np.abs(X_scaled.mean(axis=0)).max() < 1e-10
