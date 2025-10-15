# project imports
from network_security.exceptions import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.entity.artifact_entity import DataIngestionArtifact

# importing config entitiy of Data Ingestion Config
from network_security.entity.config_entity import DataIngestionConfig

# general imports
from typing import List
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os, sys
import pymongo

# dot evn loader
from dotenv import load_dotenv
load_dotenv()

"""
Steps for this data_ingestion modele:

1) Read data from MongoDB
2) Store data in feature store
3) Split the data into train and test sets
4) Store the train and test sets in the ingested directory
"""

### DEFINING CONSTANTS ###
MONGO_DB_URL = os.getenv("MONGODB_URI")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """
        Reads data from MongoDB collection and converts it to a pandas DataFrame

        Returns:
            pd.DataFrame: DataFrame containing the data from the MongoDB collection
        """
        try:
            database_name = self.data_ingestion_config.db_name
            collection_name = self.data_ingestion_config.collection_name

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns:
                df = df.drop(columns=["_id"], axis=1)
            df.replace("na", np.nan, inplace=True)
            
            return df
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_data_to_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Exports the DataFrame to the feature store path as a CSV file
        Args:
            dataframe (pd.DataFrame): The DataFrame to be exported
        """
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path

            # creating the folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size = self.data_ingestion_config.train_test_split_ratio,
            )
            logging.info("Performed train test split on the data")

            logging.info("Exited the split_data_as_train_test method of Data Ingestion class")

            dir_path = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info("Created the directory for train and test data")

            train_set.to_csv(self.data_ingestion_config.train_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_file_path, index=False, header=True)

            logging.info("Exported train and test data to their respective paths")

        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def initiate_data_ingestion(self):
        try:
            # get data from mongoDB as data frame
            dataframe = self.export_collection_as_dataframe()

            # export data to feature store
            dataframe = self.export_data_to_feature_store(dataframe)

            # dropping columns if required
            self.split_data_as_train_test(dataframe)

            # creating data ingestion artifact
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path = self.data_ingestion_config.train_file_path,
                test_file_path = self.data_ingestion_config.test_file_path
            )
            return data_ingestion_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

