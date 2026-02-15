import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


class TestLoanAPI:
    @patch("serve.app.joblib")
    def test_health_endpoint(self, mock_joblib):
        from serve.app import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()

    @patch("serve.app.joblib")
    def test_health_returns_model_status(self, mock_joblib):
        from serve.app import app
        client = TestClient(app)
        response = client.get("/health")
        data = response.json()
        assert "model_loaded" in data
