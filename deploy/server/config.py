"""Server configuration loaded from environment variables with sensible defaults."""

import os


class ServerConfig:
    """Central configuration for the REST API server.

    All values are read from environment variables at import time,
    falling back to defaults that allow local development without
    any extra setup.

    Env vars
    --------
    APP_ENV           : str   — dev | staging | prod  (default: dev)
    APP_HOST          : str   — bind address           (default: 0.0.0.0)
    APP_PORT          : int   — listen port            (default: 8000)
    MODEL_NAME        : str   — catboost | xgboost | ridge (default: catboost)
    MODEL_VERSION     : str   — artifact version tag   (default: v1)
    LOG_LEVEL         : str   — DEBUG | INFO | WARNING | ERROR (default: INFO)
    ENABLE_METRICS    : bool  — push CloudWatch metrics (default: false)
    ENABLE_DRIFT      : bool  — run drift detection    (default: false)
    PREDICTION_LOG_PATH : str — JSONL log path         (default: logs/predictions.jsonl)
    AWS_REGION        : str   — AWS region             (default: ap-southeast-1)
    """

    APP_ENV: str = os.getenv("APP_ENV", "dev")
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
    MODEL_NAME: str = os.getenv("MODEL_NAME", "catboost")
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "v1")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "false").lower() == "true"
    ENABLE_DRIFT: bool = os.getenv("ENABLE_DRIFT", "false").lower() == "true"
    PREDICTION_LOG_PATH: str = os.getenv("PREDICTION_LOG_PATH", "logs/predictions.jsonl")
    AWS_REGION: str = os.getenv("AWS_REGION", "ap-southeast-1")
