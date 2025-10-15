from network_security.components.data_ingestion import DataIngestion
from network_security.exceptions.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig
import sys


if __name__ == "__main__":
    try:
        logging.info("Entered main try block")
        logging.info("Started data ingestion")

        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)

        # initiating the data ingestion process
        logging.info("Initiating data ingestion")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)
    except Exception as e:
        raise NetworkSecurityException(e, sys)