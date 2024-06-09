from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class DataPreprocessor(ABC):
    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Preprocess the dataframe.
        
        Args:
            df: pd.DataFrame = the dataframe to preprocess.

        Returns:
            DataFrame that has been preprocessed.
        '''
        pass