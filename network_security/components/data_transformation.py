from network_security.exceptions.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.utils.main_utils.utils import save_numpy_array, save_object
from network_security.constants.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS
from network_security.entity.config_entity import DataTransformationConfig
from network_security.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np
import os, sys


class DataTransformation:
    def __init__(self,
                 data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
            try:
                self.data_validation_artifact = data_validation_artifact
                self.data_transformation_config = data_transformation_config
            except Exception as e:
                raise NetworkSecurityException(e, sys)
            
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def get_knn_transformation_object(cls) -> Pipeline:
        """
        Initialize the KNN imputer object with the parameters defined in the constants file
        """
        logging.info("Inside get_transformation_object method of DataTransformation class")
        try:
            knn_imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            processor: Pipeline = Pipeline([("imputer", knn_imputer)])
            return processor
        except Exception as e:
            raise NetworkSecurityException(e, sys)
            
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Starting data transformation")
        try:
            # reading train and test data
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # removing target variable
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            # replace -1s in target feature to 0 for better classification
            target_feature_train_df.replace(-1, 0, inplace=True)
            target_feature_test_df.replace(-1, 0, inplace=True)

            # implmenting the KNN imputer
            knn_preprocessor = self.get_knn_transformation_object()
            knn_processor_obj = knn_preprocessor.fit(input_feature_train_df)
            transformed_input_feature_train_df = knn_processor_obj.transform(input_feature_train_df)
            transformed_input_feature_test_df = knn_processor_obj.transform(input_feature_test_df)

            # combining input and target features for both train and tests datasets
            train_nparray = np.c_[transformed_input_feature_train_df, np.array(target_feature_train_df)]
            test_nparray = np.c_[transformed_input_feature_test_df, np.array(target_feature_test_df)]

            # saving the numpty arrays and the object into their respective paths
            save_numpy_array(self.data_transformation_config.transformed_train_file_path, array=train_nparray)
            save_numpy_array(self.data_transformation_config.transformed_test_file_path, array=test_nparray)
            save_object(self.data_transformation_config.transformed_object_file_path, obj=knn_processor_obj)

            # preparing artifacts
            data_transformation_artifact = DataTransformationArtifact(
                transformation_object_path = self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path = self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path= self.data_transformation_config.transformed_test_file_path
            )
            logging.info("Data transformation completed")
            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
