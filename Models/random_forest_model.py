import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas_ta
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from Models.base_model import ModelTrainer
from Utils.logger_config import configure_logger

class RandomForestTrainer(ModelTrainer):
    def __init__(self):
        self.logger = configure_logger(self.__class__.__name__)
        self.model = None
        self.X_test = None
        self.y_test = None

    def train(self, df: pd.DataFrame, target_column: str) -> RandomForestClassifier: # return a RandomForestClassifier
        self.logger.info("Training Random Forest Classifier Model")
        try:
            # selection of features and target
            X = df.drop(target_column, axis=1) # remove the target column
            y = df[target_column] # only the target column
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            
            self.X_test = X_test
            self.y_test = y_test

            rf_classifier = RandomForestClassifier(random_state=42)

            # parameter grid for parameter tuning
            parameters = {
                'n_estimators': [200, 300],
                'max_depth': [10, 30],
                'min_samples_split': [2, 10],
                'min_samples_leaf': [1, 4]
            }

            # Use GridSearchCV for hyperparameter tuning
            grid_search = GridSearchCV(estimator=rf_classifier, param_grid=parameters, cv=5, n_jobs=-1, verbose=2)
            grid_search.fit(X_train, y_train)

            # Make predictions with the best estimator
            self.model = grid_search.best_estimator_
            self.logger.info('Random Forest Classifier model trained.')
            return self.model
        except Exception as e:
            self.logger.error(f"Error training Random Forest Classifier model: {str(e)}")
            raise e

    def predict(self, model: RandomForestClassifier, features: pd.DataFrame) ->np.ndarray:
        return model.predict(features)

    def evaluate(self, model:RandomForestClassifier) -> dict:
        self.logger.info('Evaluating Random Forest Classifier model...')
        if self.X_test is None or self.y_test is None:
            raise ValueError("Test was not found. Make sure to call the train() function before the evaluate() function.")

        predictions = model.predict(self.X_test)
        report = classification_report(self.y_test, predictions, output_dict=True)
        self.logger.info(classification_report(self.y_test, predictions))
        return {'classification_report': report}
