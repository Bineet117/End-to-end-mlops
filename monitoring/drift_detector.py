import numpy as np
import pandas as pd
from scipy import stats
from loggings.logger import get_logger
from src.utils.config_loader import ConfigLoader
import joblib

logger = get_logger("monitoring.drift_detector")


class DriftDetector:
    def __init__(self):
        self.config_loader = ConfigLoader()

    def load_reference_data(self, config):
        """Load the reference (training) data."""
        ref_path = config.get("drift", {}).get("reference_data_path")
        if ref_path:
            return joblib.load(ref_path)
        return None

    def detect_numerical_drift(self, reference, current, feature, threshold):
        """Detect drift in numerical features using KS test."""
        statistic, p_value = stats.ks_2samp(
            reference[feature], current[feature]
        )
        drifted = p_value < threshold
        return {
            "feature": feature,
            "test": "KS Test",
            "statistic": round(statistic, 4),
            "p_value": round(p_value, 4),
            "threshold": threshold,
            "drifted": drifted,
        }

    def detect_categorical_drift(self, reference, current, feature, threshold):
        """Detect drift in categorical features using Chi-squared test."""
        ref_counts = reference[feature].value_counts()
        cur_counts = current[feature].value_counts()

        # Align categories
        all_categories = set(ref_counts.index) | set(cur_counts.index)
        ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
        cur_aligned = [cur_counts.get(cat, 0) for cat in all_categories]

        if sum(cur_aligned) == 0:
            return {
                "feature": feature,
                "test": "Chi-squared",
                "statistic": 0,
                "p_value": 1.0,
                "threshold": threshold,
                "drifted": False,
            }

        statistic, p_value = stats.chisquare(cur_aligned, f_exp=ref_aligned)
        drifted = p_value < threshold
        return {
            "feature": feature,
            "test": "Chi-squared",
            "statistic": round(statistic, 4),
            "p_value": round(p_value, 4),
            "threshold": threshold,
            "drifted": drifted,
        }

    def run_drift_check(self, current_data):
        """Run drift detection on incoming data against reference."""
        config = self.config_loader.load("monitor")
        threshold = config.get("drift", {}).get("threshold", 0.05)
        features_to_monitor = config.get("drift", {}).get(
            "features_to_monitor", []
        )

        reference_data = self.load_reference_data(config)
        if reference_data is None:
            logger.warning("No reference data found. Skipping drift check.")
            return []

        results = []
        for feature in features_to_monitor:
            if feature not in current_data.columns:
                logger.warning(f"Feature '{feature}' not found in current data")
                continue

            if current_data[feature].dtype in [np.float64, np.int64, float, int]:
                result = self.detect_numerical_drift(
                    reference_data, current_data, feature, threshold
                )
            else:
                result = self.detect_categorical_drift(
                    reference_data, current_data, feature, threshold
                )
            results.append(result)

            status = "DRIFT DETECTED" if result["drifted"] else "OK"
            logger.info(
                f"  {feature}: {status} "
                f"(p_value={result['p_value']}, threshold={threshold})"
            )

        drift_count = sum(1 for r in results if r["drifted"])
        logger.info(
            f"Drift check complete: {drift_count}/{len(results)} features drifted"
        )
        return results
