from network_security.exceptions.exception import NetworkSecurityException
from network_security.logging.logger import logging

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

import pandas as pd
import numpy as np
import os, sys
import yaml
import dill
import pickle


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    

def write_yaml_file(file_path: str, 
                    content: object, 
                    replace: bool = False
                    ) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    

def save_numpy_array(file_path: str, array: np.array) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file:
            np.save(file, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    

def load_numpy_array(file_path: str) -> np.array:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")
        with open(file_path, "rb") as file:
            return np.load(file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def save_object(file_path: str, obj: object) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with open(file_path, "wb") as file:
            pickle.dump(obj, file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    

def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")
        with open(file_path, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def evaluate_models(X_train, y_train, 
                    X_test, y_test,
                    models: dict, 
                    params: dict
                    ) -> dict:
    try:
        report = {}
        
        for model_name, model in models.items():
            # Get parameters for this model
            param = params[model_name]

            # Perform GridSearch
            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train, y_train)

            # Set best parameters and retrain
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store test score in report
            report[model_name] = test_model_score
            
        return report
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
