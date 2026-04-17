"""Create or update a SageMaker real-time endpoint for the taxi fare model.

Usage:
    python deploy/sagemaker/deploy_endpoint.py

All AWS-specific values are read from environment variables so that no
secrets or account IDs are hardcoded.  Placeholders are documented below.

Required env vars (replace <REPLACE> with real values when deploying):
    SAGEMAKER_ROLE        — IAM role ARN, e.g. arn:aws:iam::<REPLACE>:role/SageMakerExecRole
    MODEL_S3_URI          — S3 URI of model.tar.gz, e.g. s3://<REPLACE>/models/taxi-fare/model.tar.gz
    ECR_IMAGE_URI         — ECR image for inference container,
                            e.g. <REPLACE>.dkr.ecr.ap-southeast-1.amazonaws.com/taxi-fare-api:latest

Optional env vars:
    ENDPOINT_NAME         — default: taxi-fare-endpoint
    INSTANCE_TYPE         — default: ml.m5.large
    INSTANCE_COUNT        — default: 1
    AWS_REGION            — default: ap-southeast-1
"""

from __future__ import annotations

import logging
import os
import sys
import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _env(key: str, default: str | None = None) -> str:
    val = os.getenv(key, default)
    if val is None:
        logger.error("Required env var %s is not set", key)
        sys.exit(1)
    return val


def deploy() -> None:
    """Create or update the SageMaker endpoint."""
    import boto3

    region = _env("AWS_REGION", "ap-southeast-1")
    sm = boto3.client("sagemaker", region_name=region)

    role = _env("SAGEMAKER_ROLE")  # arn:aws:iam::<REPLACE>:role/SageMakerExecRole
    model_s3 = _env("MODEL_S3_URI")  # s3://<REPLACE>/models/taxi-fare/model.tar.gz
    image_uri = _env("ECR_IMAGE_URI")  # <REPLACE>.dkr.ecr.ap-southeast-1.amazonaws.com/taxi-fare-api:latest
    endpoint_name = _env("ENDPOINT_NAME", "taxi-fare-endpoint")
    instance_type = _env("INSTANCE_TYPE", "ml.m5.large")
    instance_count = int(_env("INSTANCE_COUNT", "1"))

    timestamp = int(time.time())
    model_name = f"taxi-fare-model-{timestamp}"
    config_name = f"taxi-fare-config-{timestamp}"

    # ---- 1. Create Model ---------------------------------------------------
    logger.info("Creating model %s", model_name)
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_s3,
            "Environment": {
                "MODEL_NAME": "catboost",
                "MODEL_VERSION": "v1",
            },
        },
        ExecutionRoleArn=role,
    )

    # ---- 2. Create Endpoint Config -----------------------------------------
    logger.info("Creating endpoint config %s", config_name)
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "primary",
                "ModelName": model_name,
                "InstanceType": instance_type,
                "InitialInstanceCount": instance_count,
                "InitialVariantWeight": 1.0,
            }
        ],
    )

    # ---- 3. Create or Update Endpoint --------------------------------------
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        logger.info("Updating existing endpoint %s", endpoint_name)
        sm.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )
    except sm.exceptions.ClientError:
        logger.info("Creating new endpoint %s", endpoint_name)
        sm.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )

    # ---- 4. Wait -----------------------------------------------------------
    logger.info("Waiting for endpoint to be InService ...")
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(
        EndpointName=endpoint_name,
        WaiterConfig={"Delay": 30, "MaxAttempts": 60},
    )
    logger.info("Endpoint %s is InService", endpoint_name)


if __name__ == "__main__":
    deploy()
