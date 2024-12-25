from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class ModelTrainer(ABC): # abstract base class
    @abstractmethod
    def train(self, df: pd.DataFrame, target_column: str):
        '''
        Train the model with the provided dataframe. Must be preprocessed to fit the model first.

        Args:
            df: (pd.DataFrame) = DataFrame that contain the features and target columns.

        Returns:
            Subclasses will return the best fit model. After parameterization.
        '''
        pass

    @abstractmethod
    def predict(self, model, features: pd.DataFrame) -> np.ndarray:
        '''
        Predict the target value for the given features using the trained, best fit model.

        Args:
            model = The trained model used for predictions
            features: pd.DataFrame = The DataFrame containing the feature values

        Returns:
            np.ndarray: Return an array containing the predicted values.
        '''
        pass

    @abstractmethod
    def evaluate(self, model, df: pd.DataFrame, target_column: str) -> dict:
        '''
        Evaluate the model's performance on the provided preprocessed dataframe and the target column.

        Args:
            model = The trained best fit model that needs to be evaluated
            df: (pd.DataFrame) = DataFrame that contain the features and target columns.
            target_column = The name of the target column in the dataframe.

        Returns:
            Subclasses will return the best fit model. After parameterization.
        '''
        pass