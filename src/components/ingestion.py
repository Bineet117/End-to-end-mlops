from google.cloud import storage
from src.utils.config_loader import ConfigLoader
from loggings.logger import get_logger
import os
import sys

module_name = sys.modules[__name__].__spec__.name if __spec__ else __name__
logger = get_logger(module_name)


configloader = ConfigLoader()

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Construct a blob (an object)
    blob = bucket.blob(source_blob_name)
    
    print(f"Downloading {source_blob_name} to {destination_file_name}...")

    # Download blob to local file
    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")


configs = configloader.load("gcp")

# Replace these with your actual values
logger.info("import about to start everthing")
MY_BUCKET = configs.get("gcs", {}).get("bucket_name")
GCS_FILE_PATH = configs.get("gcs", {}).get("gcs_raw_file_path")
LOCAL_PATH = "data/raw/raw_data.csv"
logger.info("imported everthing")

try:
    download_blob(MY_BUCKET, GCS_FILE_PATH, LOCAL_PATH)
except Exception as e:
    print(f"An error occurred: {e}")


