from network_security.constants.training_pipeline import MODEL_TRAINER_TRAINED_MODEL_FILE_NAME
from network_security.exceptions.exception import NetworkSecurityException
from network_security.logging.logger import logging
import os, sys


class NetworkModel:
    def __init__(self, preprocessor: object, model: object):
        try:
            self.processor = preprocessor
            self.model = model
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def predict(self, X):
        try:
            X_transform = self.preprocessor.transform(X)
            y_hat = self.model.predict(X_transform)
            return y_hat
        except Exception as e:
            raise NetworkSecurityException(e, sys)