from network_security.components.data_ingestion import DataIngestion
from network_security.components.data_validation import DataValidation
from network_security.exceptions.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.entity.config_entity import DataIngestionConfig, DataValidationConfig, TrainingPipelineConfig
import sys


if __name__ == "__main__":
    try:
        logging.info("Entered main try block")
        logging.info("Started data ingestion")

        training_pipeline_config = TrainingPipelineConfig()

        # data ingestion configuration
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)

        # initiating the data ingestion process
        logging.info("Initiating data ingestion")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(f"Data Ingestion Artifact: \n{data_ingestion_artifact} \n")
        logging.info("Data ingestion completed")

        # data validation configuration
        data_validation_config = DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation = DataValidation(
            data_ingestion_artifact = data_ingestion_artifact,
            data_validation_config = data_validation_config
        )

        # initiating data validation
        logging.info("Initiating data validation")
        
        data_validation_artifact = data_validation.initiate_data_validation()
        print(f"Data Validation Artifact: \n{data_validation_artifact} \n")
    except Exception as e:
        raise NetworkSecurityException(e, sys)