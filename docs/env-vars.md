# Environment Variables

All configuration is controlled through environment variables with sensible defaults for local development.

## Application

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | `dev` | Environment: `dev`, `staging`, `prod` |
| `APP_HOST` | `0.0.0.0` | Server bind address |
| `APP_PORT` | `8000` | Server listen port |
| `MODEL_NAME` | `catboost` | Model to load: `catboost`, `xgboost`, `ridge` |
| `MODEL_VERSION` | `v1` | Model version tag |
| `LOG_LEVEL` | `INFO` | Python log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

## Monitoring

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_METRICS` | `false` | Push metrics to CloudWatch (`true`/`false`) |
| `ENABLE_DRIFT` | `false` | Enable drift detection (`true`/`false`) |
| `PREDICTION_LOG_PATH` | `logs/predictions.jsonl` | Path for structured prediction log |

## AWS

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_REGION` | `ap-southeast-1` | AWS region for CloudWatch and SageMaker |

## SageMaker Deployment (deploy_endpoint.py)

These are required only when running the SageMaker deployment script.

| Variable | Default | Description |
|----------|---------|-------------|
| `SAGEMAKER_ROLE` | (required) | IAM role ARN: `arn:aws:iam::<REPLACE>:role/SageMakerExecRole` |
| `MODEL_S3_URI` | (required) | S3 model archive: `s3://<REPLACE>/models/taxi-fare/model.tar.gz` |
| `ECR_IMAGE_URI` | (required) | ECR inference image: `<REPLACE>.dkr.ecr.ap-southeast-1.amazonaws.com/taxi-fare-api:latest` |
| `ENDPOINT_NAME` | `taxi-fare-endpoint` | SageMaker endpoint name |
| `INSTANCE_TYPE` | `ml.m5.large` | SageMaker instance type |
| `INSTANCE_COUNT` | `1` | Number of instances behind the endpoint |

## GitHub Actions Secrets

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM access key for ECR push |
| `AWS_SECRET_ACCESS_KEY` | IAM secret key for ECR push |
| `AWS_ACCOUNT_ID` | 12-digit AWS account ID |
