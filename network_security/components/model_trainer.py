from network_security.exceptions.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.utils.ml_utils.model.estimator import NetworkModel
from network_security.utils.ml_utils.metric.classification_metric import get_classification_score
import network_security.constants.training_pipeline as tp
from network_security.entity.config_entity import (
    DataTransformationConfig,
    ModelTrainerConfig
)
from network_security.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from network_security.utils.main_utils.utils import (
    save_object, 
    load_object, 
    load_numpy_array,
    evaluate_models
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from xgboost import XGBClassifier
from sklearn.metrics import r2_score

import pandas as pd
import numpy as np
import os, sys


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        


    def train_model(self, X_train, y_train, X_test, y_test) -> object:
        try:
            logging.info("Training the model")
            
            # intializing models
            models = {
                "LogisticRegression": LogisticRegression(max_iter=1000),
                "KNeighborsClassifier": KNeighborsClassifier(),
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "RandomForestClassifier": RandomForestClassifier(),
                "AdaBoostClassifier": AdaBoostClassifier(),
                "GradientBoostingClassifier": GradientBoostingClassifier(),
                "XGBClassifier": XGBClassifier()
            }

            # defining parameters for hyperparameter tuning
            params = {
                "DecisionTreeClassifier": {
                    "criterion": ['gini', 'entropy'],
                    # "splitter": ['best', 'random'],
                    # "max_features": ['sqrt', 'log2', None],
                    # "max_depth": [3, 5, 10, 15, 20, None]
                },
                "RandomForestClassifier": {
                    # "criterion": ['gini', 'entropy', "log_loss"],
                    # "max_features": ['sqrt', 'log2', None],
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 5, 10, 15, 20, None]
                },
                "GradientBoostingClassifier": {
                    "loss": ['log_loss', 'exponential'],
                    # "learning_rate": [0.1, 0.01, 0.001, 0.05],
                    # "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # "criterion": ['friedman_mse', 'squared_error'],
                    # "max_features": ['sqrt', 'log2', None],
                    "n_estimators": [50, 100, 200],
                    # "max_depth": [3, 5, 10]
                },
                "LogisticRegression": {
                    # "penalty": ['l1', 'l2', 'elasticnet', None],
                    # "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                    # "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                    "max_iter": [100, 200, 500]
                },
                "KNeighborsClassifier": {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    # "weights": ['uniform', 'distance'],
                    # "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    # "p": [1, 2],
                    # "leaf_size": [10, 20, 30, 40, 50]
                },
                "AdaBoostClassifier": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.1, 0.01, 0.001, 0.05, 1.0],
                    # "algorithm": ['SAMME', 'SAMME.R']
                },
                "XGBClassifier": {
                    "n_estimators": [50, 100, 200],
                    # "learning_rate": [0.1, 0.01, 0.001, 0.05],
                    # "max_depth": [3, 5, 7, 9],
                    # "gamma": [0, 0.1, 0.2],
                    # "subsample": [0.6, 0.7, 0.8, 0.9]
                }
            }

            model_report: dict = evaluate_models(
                X_train= X_train,
                y_train= y_train,
                X_test = X_test,
                y_test = y_test,
                models = models,
                params = params
            )

            # getting the best model score from the report
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = max(sorted(model_report.values()))
            best_model = models[best_model_name]

            y_train_pred = best_model.predict(X_train)
            train_classification_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

            # tracking the MLFlow


            # getting the test classification metrics
            y_test_pred = best_model.predict(X_test)
            test_classification_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            # loading the object, saving it
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformation_object_path)
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)
            
            # saving the object
            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=network_model
            )

            # saving the model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_classification_metric,
                test_metric_artifact=test_classification_metric
            )
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)





    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Initiating model trainer")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # loading the training and testing arrays
            training_array = load_numpy_array(file_path=train_file_path)
            testing_array = load_numpy_array(file_path=test_file_path)

            # splitting the training and testing arrays into input and target feature arrays
            X_train, y_train = training_array[:, :-1], training_array[:, -1]
            X_test, y_test = testing_array[:, :-1], testing_array[:, -1]

            # creating model
            model = self.train_model(X_train, y_train, X_test, y_test)

            logging.info("Model training completed")
            return model
        except Exception as e:
            raise NetworkSecurityException(e, sys)