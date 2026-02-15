from src.components.ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.preprocess import DataPreprocessor
from src.components.train import ModelTrainer
from src.components.evaluate import ModelEvaluator
from loggings.logger import get_logger
import sys

module_name = sys.modules[__name__].__spec__.name if __spec__ else __name__
logger = get_logger(module_name)


def run_training_pipeline():
    """Run the full training pipeline: Ingest → Validate → Preprocess → Train → Evaluate."""
    logger.info("=" * 60)
    logger.info("STARTING LOAN PREDICTION TRAINING PIPELINE")
    logger.info("=" * 60)

    try:
        # Step 1: Data Ingestion
        logger.info("STEP 1: Data Ingestion")
        ingestion = DataIngestion()
        df = ingestion.download_blob("gcp")
        logger.info(f"Ingested data shape: {df.shape}")

        # Step 2: Data Validation
        logger.info("STEP 2: Data Validation")
        validator = DataValidation()
        validator.validate(df, "validation")
        logger.info("Data validation complete")

        # Step 3: Data Preprocessing
        logger.info("STEP 3: Data Preprocessing")
        preprocessor = DataPreprocessor()
        df = preprocessor.preprocess(df)
        df = preprocessor.mapping(df)
        logger.info(f"Preprocessed data shape: {df.shape}")

        # Save processed data
        preprocessor.save_processed_data(df, "data/processed/processed_data.pkl")

        # Step 4: Model Training
        logger.info("STEP 4: Model Training")
        trainer = ModelTrainer()
        model, X_test, y_test = trainer.train(df)
        logger.info("Model training complete")

        # Step 5: Model Evaluation
        logger.info("STEP 5: Model Evaluation")
        evaluator = ModelEvaluator()
        metrics, passed = evaluator.evaluate(model, X_test, y_test)

        if passed:
            logger.info("Pipeline completed SUCCESSFULLY - model meets thresholds")
        else:
            logger.warning("Pipeline completed with WARNINGS - model below thresholds")

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        return metrics, passed

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    run_training_pipeline()
