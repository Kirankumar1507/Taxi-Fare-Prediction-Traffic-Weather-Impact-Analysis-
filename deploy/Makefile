# Taxi Fare Prediction API — Build & Run targets
# ================================================
# Run from project root: make -f deploy/Makefile <target>

IMAGE_NAME ?= taxi-fare-api
IMAGE_TAG  ?= latest
PORT       ?= 8000

.PHONY: build run stop test smoke clean logs

# -- Docker ---------------------------------------------------------------
build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) -f deploy/Dockerfile .

run: build
	docker run -d --name $(IMAGE_NAME) -p $(PORT):8000 \
		-e APP_ENV=dev -e LOG_LEVEL=DEBUG \
		$(IMAGE_NAME):$(IMAGE_TAG)
	@echo "API running at http://localhost:$(PORT)"

stop:
	docker stop $(IMAGE_NAME) 2>/dev/null || true
	docker rm   $(IMAGE_NAME) 2>/dev/null || true

# -- Compose --------------------------------------------------------------
up:
	docker compose -f deploy/docker-compose.yaml up --build -d

down:
	docker compose -f deploy/docker-compose.yaml down

# -- Tests ----------------------------------------------------------------
test:
	python -m pytest tests/ -v --tb=short

smoke:
	@echo "--- Health check ---"
	curl -sf http://localhost:$(PORT)/health | python -m json.tool
	@echo "\n--- Single prediction ---"
	curl -sf -X POST http://localhost:$(PORT)/predict \
		-H "Content-Type: application/json" \
		-d '{"Trip_Distance_km":19.35,"Passenger_Count":3,"Base_Fare":3.56,"Per_Km_Rate":0.8,"Per_Minute_Rate":0.32,"Trip_Duration_Minutes":53.82,"Traffic_Conditions":"Low","Weather":"Clear","Time_of_Day":"Morning","Day_of_Week":"Weekday"}' \
	| python -m json.tool
	@echo "\n--- Batch prediction ---"
	curl -sf -X POST http://localhost:$(PORT)/predict/batch \
		-H "Content-Type: application/json" \
		-d '{"trips":[{"Trip_Distance_km":5,"Passenger_Count":1,"Base_Fare":2.5,"Per_Km_Rate":1.0,"Per_Minute_Rate":0.3,"Trip_Duration_Minutes":15,"Traffic_Conditions":"Medium","Weather":"Rain","Time_of_Day":"Night","Day_of_Week":"Weekend"},{"Trip_Distance_km":10,"Passenger_Count":2,"Base_Fare":3.0,"Per_Km_Rate":0.9,"Per_Minute_Rate":0.35,"Trip_Duration_Minutes":25,"Traffic_Conditions":"High","Weather":"Clear","Time_of_Day":"Afternoon","Day_of_Week":"Weekday"}]}' \
	| python -m json.tool

# -- Cleanup --------------------------------------------------------------
clean: stop
	docker rmi $(IMAGE_NAME):$(IMAGE_TAG) 2>/dev/null || true
	rm -rf logs/predictions.jsonl

logs:
	docker logs -f $(IMAGE_NAME) 2>/dev/null || echo "Container not running"
