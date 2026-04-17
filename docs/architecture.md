# Architecture — Taxi Fare Prediction System

## System Diagram

```
                        +-------------------+
                        |   GitHub Actions   |
                        |  (CI/CD Pipeline)  |
                        +--------+----------+
                                 |
                    push to main | PR: run tests
                                 v
+----------+     +----------+    +-----------+    +-----------------+
|  Client  | --> |  FastAPI  | -> | FarePre-  | -> | CatBoost Model  |
| (REST)   |     |  Server   |    | dictor    |    | (catboost_v1)   |
+----------+     +-----+----+    +-----------+    +-----------------+
                       |
           +-----------+-----------+
           |           |           |
           v           v           v
     +---------+  +--------+  +---------+
     | Health  |  | Pred   |  | Cloud-  |
     | Check   |  | Log    |  | Watch   |
     |         |  | (JSONL)|  | Metrics |
     +---------+  +---+----+  +---------+
                      |
                      v
               +------------+
               |   Drift    |
               | Detection  |
               |   (PSI)    |
               +------+-----+
                      |
                      v
               +------------+
               |  Retrain   |
               |  Trigger   |
               +------------+
```

## Data Flow

```
Raw Input (10 fields)
    |
    v
Pydantic Validation (schemas.py)
    |
    v
FarePredictor._validate()
    |
    v
FarePredictor._build_features()
    |  - Ordinal encode 4 categoricals
    |  - Engineer 9 derived features
    |  - Produce 19-dim feature vector
    v
CatBoost.predict()
    |
    v
Response (predicted_fare, model_name, model_version)
    |
    +---> Prediction Log (JSONL)
    +---> CloudWatch Metrics (latency, count)
```

## Component Responsibilities

| Component | Location | Responsibility |
|-----------|----------|---------------|
| REST Server | `deploy/server/rest_server.py` | FastAPI app, routing, error handling |
| Schemas | `deploy/server/schemas.py` | Pydantic request/response validation |
| Config | `deploy/server/config.py` | Env var configuration |
| Health | `deploy/server/health.py` | Readiness + liveness checks |
| FarePredictor | `src/predict.py` | Model loading, feature eng, inference |
| Prediction Log | `deploy/monitoring/prediction_log.py` | JSONL audit trail |
| Metrics | `deploy/monitoring/metrics.py` | CloudWatch metrics sink |
| Drift Detection | `deploy/monitoring/drift.py` | PSI-based distribution shift detection |
| SageMaker Entry | `deploy/sagemaker/inference.py` | model_fn/input_fn/predict_fn/output_fn |
| SageMaker Deploy | `deploy/sagemaker/deploy_endpoint.py` | Endpoint create/update via boto3 |

## Deployment Targets

### Local (Docker Compose)
No AWS credentials needed. Uses the same Docker image with `APP_ENV=dev`.

### SageMaker (Production)
- Model artifacts packaged as `model.tar.gz` in S3
- Docker image pushed to ECR
- Endpoint managed via `deploy_endpoint.py` or CodeBuild

### CI/CD Pipeline
- **PR** -> `test.yml` -> pytest
- **Merge to main** -> `build-api.yml` -> build Docker, push to ECR

## Environments

| Environment | APP_ENV | Metrics | Drift | Notes |
|-------------|---------|---------|-------|-------|
| Local | dev | off | off | docker-compose, hot reload |
| Staging | staging | on | on | SageMaker, ml.m5.large |
| Production | prod | on | on | SageMaker, ml.m5.large x 2 |
