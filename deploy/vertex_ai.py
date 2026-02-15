"""
Vertex AI Deployment Pipeline

1. Uploads the trained model artifacts from GCS to Vertex AI Model Registry.
2. Deploys the registered model to a Vertex AI Endpoint using the custom container.

Usage:
    python -m deploy.vertex_ai
"""

from google.cloud import aiplatform
from src.utils.config_loader import ConfigLoader
from loggings.logger import get_logger
import time

logger = get_logger("deploy.vertex_ai")


def deploy_to_vertex_ai():
    logger.info("Starting Vertex AI deployment...")

    config_loader = ConfigLoader()
    gcp_config = config_loader.load("gcp")
    
    project_id = gcp_config.get("gcp_login", {}).get("project_id")
    region = gcp_config.get("gcp_login", {}).get("region")
    bucket_name = gcp_config.get("gcs", {}).get("bucket_name")

    if not project_id or not bucket_name:
        logger.error("Missing Project ID or Bucket Name in gcp.yaml")
        return

    # Initialize Vertex AI SDK
    aiplatform.init(project=project_id, location=region)

    # 1. Define Model Artifacts (uploaded by train.py)
    # The path must be the folder containing model.pkl, scaler.pkl, etc.
    model_artifact_uri = f"gs://{bucket_name}/models/"
    
    # 2. Define Custom Container Image
    # Ensure you have pushed this image: docker push ...
    repository = "mlops"  # Artifact Registry repo name
    image_name = "loan-prediction"
    image_tag = "latest"
    serving_container_image_uri = (
        f"{region}-docker.pkg.dev/{project_id}/{repository}/{image_name}:{image_tag}"
    )

    logger.info(f"Model Artifacts: {model_artifact_uri}")
    logger.info(f"Container Image: {serving_container_image_uri}")

    try:
        # 3. Upload Model to Registry
        logger.info("Step 1: Uploading model to Vertex AI Model Registry...")
        model = aiplatform.Model.upload(
            display_name=f"loan-prediction-{int(time.time())}",
            artifact_uri=model_artifact_uri,
            serving_container_image_uri=serving_container_image_uri,
            serving_container_predict_route="/predict",
            serving_container_health_route="/health",
            serving_container_ports=[8080],
            description="Loan prediction model validation",
        )
        logger.info(f"Model successfully uploaded: {model.resource_name}")

        # 4. Deploy to Endpoint
        logger.info("Step 2: Deploying model to Vertex AI Endpoint...")
        endpoint = model.deploy(
            deployed_model_display_name="loan-prediction-deployed",
            machine_type="n1-standard-4",
            min_replica_count=1,
            max_replica_count=1,  # Scale up for production
            traffic_percentage=100,
            sync=True,
        )
        logger.info(f"Model deployed successfully!")
        logger.info(f"Endpoint Resource Name: {endpoint.resource_name}")
        logger.info(f"Prediction URL: https://{region}-aiplatform.googleapis.com/v1/{endpoint.resource_name}:predict")

        return endpoint

    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise


if __name__ == "__main__":
    deploy_to_vertex_ai()
