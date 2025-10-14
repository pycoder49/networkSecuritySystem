from dotenv import load_dotenv
from network_security.logging.logger import logging
from network_security.exceptions.exception import NetworkSecurityException

import os, sys
import json
import certifi
import pandas as pd
import numpy as np
import pymongo

load_dotenv()

# MongoDB URL and certification for valid connection
MONGO_DB_URL = os.getenv("MONGODB_URI")
ca = certifi.where()


# implementing the ETL pipeline
class NetworkDataExtraction():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def csv_to_json_convertor(self, file_path):
        try:
            data = pd.read_csv(file_path)

            # getting rid of the index
            data.reset_index(drop=True, inplace=True)

            records = list((json.loads(data.T.to_json()).values()))
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def push_data_to_mongodb(self, records, database, collection):
        try:
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            db = self.mongo_client[database]
            collection_obj = db[collection]
            collection_obj.insert_many(records)

            return len(records)

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
if __name__ == "__main__":    
    FILE_PATH = "network_data/phisingData.csv"
    DATABASE = "aryan"
    Collection = "NetworkData"

    networkobj = NetworkDataExtraction()
    
    records = networkobj.csv_to_json_convertor(file_path=FILE_PATH)
    len_records = networkobj.push_data_to_mongodb(records=records, database=DATABASE, collection=Collection)

