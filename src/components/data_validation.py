from src.utils.config_loader import ConfigLoader
from loggings.logger import get_logger
from src.components.ingestion import DataIngestion
import sys
import numpy as np
import pandas as pd 

module_name = sys.modules[__name__].__spec__.name if __spec__ else __name__
logger = get_logger(module_name)

ingestion = DataIngestion()

class DataValidation:
    def __init__(self):
        self.configs = ConfigLoader()

    def validate(self, df, name):
        config = self.configs.load(name)
        try: 
            if len(df.columns) == 13:
                logger.warning("Number of column matched")

                logger.info(df.columns)
                data_details = config.get("data_details")
                expected_columns = (
                    [data_details["target"]] +
                    data_details["numerical"] +
                    data_details["categorical"]
                )
                logger.info(expected_columns)
                if set(expected_columns) == set(df.columns.str.strip()):
                    logger.info("The columns are same")
                else:
                    logger.warning("FAAAAAAAAAAAAAAAAAAAAAAA")

        except Exception as e:
            logger.warning(f"Number of column didn't matched : {e}")



df = ingestion.download_blob("gcp")
check = DataValidation()
check.validate(df, "validation")