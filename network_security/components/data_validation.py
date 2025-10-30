from network_security.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from network_security.entity.config_entity import DataValidationConfig
from network_security.exceptions.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.utils.main_utils.utils import read_yaml_file, write_yaml_file
from network_security.constants.training_pipeline import SCHEMA_FILE_PATH

from scipy.stats import ks_2samp    # helps with detecting drifting in data
import pandas as pd
import numpy as np
import os, sys


class DataValidation:
    """
    Static methods
    """
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    """
    Class methods start here
    """
    def __init__(self, 
                data_ingestion_artifact: DataIngestionArtifact, 
                data_validation_config: DataValidationConfig ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            logging.info("Reading train and test data for validation")

            # reading train and test data
            train_df = DataValidation.read_data(train_file_path)
            test_df = DataValidation.read_data(test_file_path)

            # validating number of columns in train dataframe
            status = self.validate_number_of_columns(train_df)
            if not status:
                logging.info("Number of columns in train dataframe are not as per schema")

            # validating number of columns in test dataframe
            status = self.validate_number_of_columns(test_df)
            if not status:
                logging.info("Number of columns in test dataframe are not as per schema")

            # checking for data drift
            status = self.detect_data_drift(base_df=train_df, current_df=test_df)
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            # saving the validated train and test data in their respective paths
            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            data_validation_artifact = DataValidationArtifact(
                validation_status = status,
                valid_train_file_path = self.data_ingestion_artifact.train_file_path,
                valid_test_file_path = self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path = None,
                invalid_test_file_path = None,
                drift_report_file_path = self.data_validation_config.drift_report_file_path
            )

            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            num_of_cols = len(self._schema_config['columns'])
            logging.info(f"Required number of columns: {num_of_cols}")
            logging.info(f"Dataframe has columns: {dataframe.shape[1]}")
            return True if dataframe.shape[1] == num_of_cols else False
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_number_of_numeric_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            # Get expected numerical columns from schema
            numerical_columns = self._schema_config['numerical_columns']
            dataframe_columns = dataframe.columns.tolist()
            
            # Check which numerical columns are present in the dataframe
            present_numerical_cols = [col for col in numerical_columns if col in dataframe_columns]
            missing_numerical_cols = [col for col in numerical_columns if col not in dataframe_columns]
            
            logging.info(f"Required number of numerical columns: {len(numerical_columns)}")
            logging.info(f"Dataframe has numerical columns: {len(present_numerical_cols)}")
            
            if missing_numerical_cols:
                logging.warning(f"Missing numerical columns: {missing_numerical_cols}")
                return False
            
            return True
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def detect_data_drift(self, 
                          base_df: pd.DataFrame,
                          current_df: pd.DataFrame,
                          threshold: float = 0.05) -> bool:
         try:
            status = True
            report = {}
            for col in base_df.columns:
                d1 = base_df[col]
                d2 = current_df[col]

                is_sample_distribution = ks_2samp(d1, d2)
                if threshold <= is_sample_distribution.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False

                report.update({
                    col: {
                        "p_value": float(is_sample_distribution.pvalue),
                        "drift_status": is_found
                    }
                })

                # creating directory for drift report file path
                drift_report_file_path = self.data_validation_config.drift_report_file_path
                dir_path = os.path.dirname(drift_report_file_path)
                os.makedirs(dir_path, exist_ok=True)

                # writing to the yaml file
                write_yaml_file(
                    file_path=drift_report_file_path,
                    content=report
                )
         except Exception as e:
             raise NetworkSecurityException(e, sys)