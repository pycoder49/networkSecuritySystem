import numpy as np
import os

"""
Training Pipeline Constants

This module contains all constants used throughout the training pipeline.
Constants are organized by functionality with descriptive prefixes.
"""


"""
Defining common constant variables for training pipeline
"""
TARGET_COLUMN: str = "Result"
PIPELINE_NAME: str = "NetworkSecurity"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "phishingData.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH:str = os.path.join("data_schema", "schema.yaml")

    
"""
Data Ingestion related constants start with DATA_INGESTION_* prefix
"""
DATA_INGESTION_COLLECTION_NAME: str = "NetworkData"
DATA_INGESTION_DATABASE_NAME: str = "aryan"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2


"""
Defining constants for data validation
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"


"""
Defining constants for data transformation
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_FILE_NAME: str = "knn_preprocessor"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "preprocessor"
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {    # knn imputer to replace nan values
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform",
}

""""
Defining constants for model trainer
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_FILE_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_ACCURACY: float = 0.7
MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD: float = 0.1