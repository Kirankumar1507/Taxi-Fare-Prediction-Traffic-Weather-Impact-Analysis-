# Runbook — Taxi Fare Prediction Service

## 1. Local Development

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-serve.txt

# Run the API server (auto-reload)
python -m deploy.server.rest_server

# Or via uvicorn directly
uvicorn deploy.server.rest_server:app --reload --port 8000

# Test endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Trip_Distance_km":19.35,"Passenger_Count":3,"Base_Fare":3.56,"Per_Km_Rate":0.8,"Per_Minute_Rate":0.32,"Trip_Duration_Minutes":53.82,"Traffic_Conditions":"Low","Weather":"Clear","Time_of_Day":"Morning","Day_of_Week":"Weekday"}'
```

## 2. Docker

```bash
# Build and run
make -f deploy/Makefile build
make -f deploy/Makefile run

# Or with docker-compose
make -f deploy/Makefile up

# Smoke test
make -f deploy/Makefile smoke

# Tear down
make -f deploy/Makefile down
```

## 3. Deploy to SageMaker

### Prerequisites
- AWS CLI configured with appropriate permissions
- ECR repository created: `taxi-fare-api`
- S3 bucket with model.tar.gz uploaded
- IAM role for SageMaker execution

### Steps

```bash
# 1. Build and push Docker image to ECR
export AWS_ACCOUNT_ID=<REPLACE>
export AWS_REGION=ap-southeast-1
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

docker build -t $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/taxi-fare-api:latest -f deploy/Dockerfile .
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/taxi-fare-api:latest

# 2. Package model artifacts and upload to S3
tar -czf model.tar.gz models/ configs/ src/ deploy/server/ deploy/monitoring/ deploy/__init__.py
aws s3 cp model.tar.gz s3://<REPLACE>/models/taxi-fare/model.tar.gz

# 3. Deploy endpoint
export SAGEMAKER_ROLE=arn:aws:iam::<REPLACE>:role/SageMakerExecRole
export MODEL_S3_URI=s3://<REPLACE>/models/taxi-fare/model.tar.gz
export ECR_IMAGE_URI=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/taxi-fare-api:latest
python deploy/sagemaker/deploy_endpoint.py
```

## 4. Rollback

### Docker / ECS rollback
```bash
# Redeploy the previous image tag
docker pull $ECR_REGISTRY/taxi-fare-api:<previous-tag>
# Update the ECS service or docker-compose to use the previous tag
```

### SageMaker rollback
```bash
# Update the endpoint to use the previous endpoint configuration
aws sagemaker update-endpoint \
  --endpoint-name taxi-fare-endpoint \
  --endpoint-config-name <previous-config-name>
```

## 5. Retrain (on drift)

```bash
# 1. Check drift status
python -m deploy.monitoring.drift

# 2. If significant drift detected:
python -m src.preprocess   # re-run with fresh data
python -m src.train        # retrain all models
python -m src.evaluate     # validate new champion

# 3. Re-package and deploy (repeat step 3 from Deploy)
```

## 6. Debug Common Issues

### Model not loading (503 on /health)
- Check `MODEL_NAME` env var matches an available model file
- Verify model files exist in `models/` directory
- Check container logs: `docker logs taxi-fare-api`

### High latency
- Check batch sizes (reduce from 100)
- Review CloudWatch `InferenceLatencyMs` metric
- Consider instance type upgrade for SageMaker

### Drift detected
- Review `deploy/monitoring/drift.py` output
- Check if input distribution has genuinely shifted
- Retrain if PSI > 0.25 on key features

### Prediction log not writing
- Verify `PREDICTION_LOG_PATH` points to a writable directory
- Check disk space in the container
