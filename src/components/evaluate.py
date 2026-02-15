import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from src.utils.config_loader import ConfigLoader
from loggings.logger import get_logger
import sys

module_name = sys.modules[__name__].__spec__.name if __spec__ else __name__
logger = get_logger(module_name)


class ModelEvaluator:
    def __init__(self):
        self.config_loader = ConfigLoader()

    def evaluate(self, model, X_test, y_test):
        """Evaluate model and return metrics."""
        config = self.config_loader.load("training")

        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
        }

        logger.info("=" * 50)
        logger.info("MODEL EVALUATION RESULTS")
        logger.info("=" * 50)
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")

        # Classification report
        report = classification_report(y_test, y_pred)
        logger.info(f"\nClassification Report:\n{report}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")

        # Check against thresholds
        eval_config = config.get("evaluation", {})
        min_accuracy = eval_config.get("min_accuracy", 0.80)
        min_f1 = eval_config.get("min_f1_score", 0.75)

        passed = True
        if metrics["accuracy"] < min_accuracy:
            logger.warning(
                f"Accuracy {metrics['accuracy']:.4f} below "
                f"threshold {min_accuracy}"
            )
            passed = False
        if metrics["f1_score"] < min_f1:
            logger.warning(
                f"F1 Score {metrics['f1_score']:.4f} below "
                f"threshold {min_f1}"
            )
            passed = False

        if passed:
            logger.info("Model PASSED evaluation thresholds")
        else:
            logger.warning("Model FAILED evaluation thresholds")

        return metrics, passed
