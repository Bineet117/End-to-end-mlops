import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestClassifier
from src.components.evaluate import ModelEvaluator


class TestModelEvaluator:
    def setup_method(self):
        self.evaluator = ModelEvaluator()

    @patch.object(ModelEvaluator, '__init__', lambda self: None)
    def test_evaluate_returns_metrics(self):
        evaluator = ModelEvaluator()
        evaluator.config_loader = MagicMock()
        evaluator.config_loader.load.return_value = {
            "evaluation": {"min_accuracy": 0.5, "min_f1_score": 0.5}
        }

        # Create a simple trained model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y)

        metrics, passed = evaluator.evaluate(model, X, y)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert isinstance(passed, bool)

    @patch.object(ModelEvaluator, '__init__', lambda self: None)
    def test_evaluate_passes_with_good_model(self):
        evaluator = ModelEvaluator()
        evaluator.config_loader = MagicMock()
        evaluator.config_loader.load.return_value = {
            "evaluation": {"min_accuracy": 0.0, "min_f1_score": 0.0}
        }

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.rand(50, 3)
        y = np.random.randint(0, 2, 50)
        model.fit(X, y)

        _, passed = evaluator.evaluate(model, X, y)
        assert passed is True
