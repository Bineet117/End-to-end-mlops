import pandas as pd
import numpy as np
import joblib
import os
from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.utils.config_loader import ConfigLoader
from loggings.logger import get_logger
import sys

module_name = sys.modules[__name__].__spec__.name if __spec__ else __name__
logger = get_logger(module_name)


class ModelTrainer:
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def upload_to_gcs(self, local_path, gcs_path):
        """Upload a local file to GCS bucket."""
        gcp_config = self.config_loader.load("gcp")
        bucket_name = gcp_config.get("gcs", {}).get("bucket_name")

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        logger.info(f"Uploaded {local_path} → gs://{bucket_name}/{gcs_path}")

    def encode_features(self, df):
        """Encode categorical features using LabelEncoder."""
        categorical_cols = ["education", "self_employed"]
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
            logger.info(f"Encoded column: {col}")
        return df

    def scale_features(self, X):
        """Scale numerical features using StandardScaler."""
        X_scaled = self.scaler.fit_transform(X)
        logger.info("Features scaled with StandardScaler")
        return X_scaled

    def split_data(self, df, config):
        """Split data into train and test sets."""
        test_size = config.get("model", {}).get("test_size", 0.2)
        random_state = config.get("model", {}).get("params", {}).get(
            "random_state", 42
        )

        X = df.drop("loan_status", axis=1)
        y = df["loan_status"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(
            f"Data split: train={X_train.shape[0]}, test={X_test.shape[0]}"
        )
        return X_train, X_test, y_train, y_test

    def train(self, df):
        """Train the model end-to-end."""
        config = self.config_loader.load("training")

        # Encode categorical features
        df = self.encode_features(df)

        # Split data
        X_train, X_test, y_train, y_test = self.split_data(df, config)

        # Scale features
        X_train_scaled = self.scale_features(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Get model params from config
        model_params = config.get("model", {}).get("params", {})
        n_estimators = model_params.get("n_estimators", 100)
        max_depth = model_params.get("max_depth", 10)
        random_state = model_params.get("random_state", 42)

        # Train RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        self.model.fit(X_train_scaled, y_train)
        logger.info(
            f"Model trained: RandomForestClassifier "
            f"(n_estimators={n_estimators}, max_depth={max_depth})"
        )

        # Save artifacts locally
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, "models/model.pkl")
        joblib.dump(self.scaler, "models/scaler.pkl")
        joblib.dump(self.label_encoders, "models/label_encoders.pkl")
        logger.info("Model and artifacts saved locally to models/")

        # Upload artifacts to GCS
        self.upload_to_gcs("models/model.pkl", "models/model.pkl")
        self.upload_to_gcs("models/scaler.pkl", "models/scaler.pkl")
        self.upload_to_gcs("models/label_encoders.pkl", "models/label_encoders.pkl")
        logger.info("All artifacts uploaded to GCS")

        # Save test data for evaluation
        os.makedirs("data/processed", exist_ok=True)
        joblib.dump(
            (X_test_scaled, y_test, X_test.columns.tolist()),
            "data/processed/test_data.pkl",
        )
        logger.info("Test data saved for evaluation")

        return self.model, X_test_scaled, y_test
