import pandas as pd
import numpy as np
import joblib
from loggings.logger import get_logger
import sys

module_name = sys.modules[__name__].__spec__.name if __spec__ else __name__
logger = get_logger(module_name)


class DataPreprocessor:
    def __init__(self):
        pass

    def preprocess(self, df):
        logger.info("Starting data preprocessing")
        df.drop("loan_id", axis=1, inplace=True)
        # removing extra space from each column name
        df.columns = [cols.strip() for cols in df.columns]
        df = df[df["residential_assets_value"] > 0]
        # creating new column `total_asset_value`
        df["total_asset_value"] = (
            df["residential_assets_value"]
            + df["commercial_assets_value"]
            + df["luxury_assets_value"]
            + df["bank_asset_value"]
        )
        # dropping these columns
        df = df.drop(
            columns=[
                "residential_assets_value",
                "commercial_assets_value",
                "luxury_assets_value",
                "bank_asset_value",
            ],
            axis=1,
        )
        logger.info(f"Preprocessing complete. Shape: {df.shape}")
        return df

    def mapping(self, data):
        logger.info("Starting data mapping/encoding")
        # removing space from the data and mapping
        data["education"] = data["education"].apply(lambda x: x.strip())

        # removing space from the data and mapping
        data["self_employed"] = data["self_employed"].apply(lambda x: x.strip())

        # removing space from the data and mapping
        data["loan_status"] = data["loan_status"].apply(lambda x: x.strip())
        data["loan_status"] = data["loan_status"].map(
            {"Approved": 1, "Rejected": 0}
        )
        logger.info("Data mapping complete")
        return data

    def save_processed_data(self, data, DATA_PATH):
        joblib.dump(data, DATA_PATH)
        logger.info(f"Processed data saved to {DATA_PATH}")
