from google.cloud import storage
from src.utils.config_loader import ConfigLoader
from loggings.logger import get_logger
import os
import sys

import pandas as pd

module_name = sys.modules[__name__].__spec__.name if __spec__ else __name__
logger = get_logger(module_name)

class DataIngestion:
    def __init__(self):
        self.configloader = ConfigLoader()

    def fetch_downloaded_data(self, destination_file_name):
        try:
            raw_data = pd.read_csv(destination_file_name)
            return raw_data
        except Exception as e:
            logger.warning(f"{e}")

    def download_blob(self, name):
        configs = self.configloader.load(name)

        bucket_name = configs.get("gcs", {}).get("bucket_name")
        source_blob_name = configs.get("gcs", {}).get("gcs_raw_file_path")
        destination_file_name = "data/raw/raw_data.csv"

        """Downloads a blob from the bucket."""
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # Construct a blob (an object)
        blob = bucket.blob(source_blob_name)

        # Download blob to local file
        blob.download_to_filename(destination_file_name)

        df = self.fetch_downloaded_data(destination_file_name)
        print(df.head())
        return df 



# ingestion = DataIngestion()
# ingestion.download_blob("gcp")


