import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas_ta
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from ml.base_trainer import ModelTrainer
from utils.logger_config import configure_logger, shared_log_stream

class RandomForestTrainer(ModelTrainer):
    def __init__(
            self,
            test_size=0.25,
            random_state=42,
            n_estimators=None,
            max_depth=None,
            min_samples_split=None,
            min_samples_leaf=None,
            grid_search_cv=3,
            grid_search_n_jobs=-1,
            grid_search_verbose=1,
    ):
        """
        Initialize the RandomForestTrainer with configurable parameters.

        Args:
            test_size (float): Test size for train-test split (default: 0.25).
            random_state (int): Random state for reproducibility (default: 42).
            n_estimators (list): List of n_estimators for GridSearchCV (default: [100, 200, 300]).
            max_depth (list): List of max_depth values for GridSearchCV (default: [None, 10, 20]).
            min_samples_split (list): List of min_samples_split for GridSearchCV (default: [2, 5, 10]).
            min_samples_leaf (list): List of min_samples_leaf for GridSearchCV (default: [1, 2, 4]).
            grid_search_cv (int): Number of cross-validation folds for GridSearchCV (default: 3).
            grid_search_n_jobs (int): Number of jobs for GridSearchCV (default: -1 for all CPUs).
            grid_search_verbose (int): Verbosity level for GridSearchCV (default: 1).
        """

        self.logger = configure_logger(self.__class__.__name__, shared_log_stream)
        self.model = None
        self.X_test = None
        self.y_test = None

        # Parameters
        # Store parameters
        self.test_size = test_size
        self.random_state = random_state
        self.n_estimators = n_estimators or [100, 200, 300]
        self.max_depth = max_depth or [None, 10, 20]
        self.min_samples_split = min_samples_split or [2, 5, 10]
        self.min_samples_leaf = min_samples_leaf or [1, 2, 4]
        self.grid_search_cv = grid_search_cv
        self.grid_search_n_jobs = grid_search_n_jobs
        self.grid_search_verbose = grid_search_verbose

    def train(self, df: pd.DataFrame, target_column: str) -> RandomForestClassifier: # return a RandomForestClassifier
        """
        Train the Random Forest Classifier using the given parameters.

        Args:
            df (pd.DataFrame): Input dataset.
            target_column (str): Column name for the target variable.

        Returns:
            RandomForestClassifier: Trained Random Forest model.
        """
        self.logger.info("Training Random Forest Classifier Model")

        try:
            # selection of features and target
            X = df.drop(target_column, axis=1) # remove the target column
            y = df[target_column] # only the target column
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                test_size=self.test_size,
                                                                random_state=self.random_state)
            
            self.X_test = X_test
            self.y_test = y_test

            rf_classifier = RandomForestClassifier(random_state=self.random_state)

            # parameter grid for parameter tuning
            parameters = {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
            }

            # Use GridSearchCV for hyperparameter tuning
            grid_search = GridSearchCV(estimator=rf_classifier,
                                       param_grid=parameters,
                                       cv=self.grid_search_cv,
                                       n_jobs=self.grid_search_n_jobs,
                                       verbose=self.grid_search_verbose)
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
        self.logger.info(f'\n{classification_report(self.y_test, predictions)}')
        return report
