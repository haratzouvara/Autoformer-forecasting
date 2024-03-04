import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import List, Optional, Union, Tuple


class ReadData:
    def __init__(self):
        pass

    @staticmethod
    def read(path: str, features: Optional[Union[List[str], None]],
             target: List[str], dates: List[str]) -> [pd.DataFrame, pd.DataFrame]:
        """
        Reads data from a CSV file, performs preprocessing, and returns a DataFrame.

        Args:
            path: Path to the CSV file.
            features: List of feature names. If provided, features will be selected along with the target.
                                             If empty or None, only the target column will be selected.
            target: Name of the target column.
            dates: Name of the column containing dates.

        Returns:
            Processed DataFrame containing selected features and the target column.
                          The target column is placed as the last column.

        """
        if not isinstance(features, List):
            raise TypeError("Features must be a list of strings.")

        if not isinstance(target, List):
            raise TypeError("Target must be a list of strings.")

        if not isinstance(dates, List):
            raise TypeError("Dates must be a list of strings.")

        data = pd.read_csv(path, parse_dates=True, index_col=dates)
        if features != target:
            data = data[features + target]
        else:
            data = data[target]
        data = data.sort_index()
        data = data.ffill().bfill()

        if len(features) > 0:
            data = data[[col for col in data.columns.values if col != target[0]] + target]

        time_features = pd.DataFrame([])
        time_features['hour'] = data.index.hour / 24
        time_features['dayofweek'] = data.index.dayofweek / 6
        time_features['month'] = data.index.month / 12
        return data, time_features.values


class ScaleData:
    def __init__(self):

        pass
        """
        Initializes a scaler based on the specified method.

        """

    @staticmethod
    def scale_train(data: pd.DataFrame, method: str) -> Tuple[np.ndarray, MinMaxScaler | StandardScaler]:
        """

        Scales the input training data based on the specified method.

        Args:
            data: Training data to be scaled.
            method: The scaling method. 'minmax' for Min-Max scaling, 'standard' for Standard scaling.

        Returns:
            Scaled training data and the corresponding scaler.
        """

        if method == 'minmax':
            scaler = MinMaxScaler(feature_range=(0, 1))
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError("Invalid scaler type. Supported types are 'minmax' and 'standard'."
                             " You must provide either a scaler instance or a valid scaler type.")

        data_train_scaled = scaler.fit_transform(data)
        return data_train_scaled, scaler

    @staticmethod
    def scale_test(data: pd.DataFrame, scaler: MinMaxScaler | StandardScaler) \
            -> Tuple[np.ndarray, MinMaxScaler | StandardScaler]:

        """
        Scales the input test data using a pre-trained scaler.

        Args:
            data: Test data to be scaled.
            scaler: Pre-trained scaler.

        Returns:
            Scaled test data and the corresponding scaler.

        """

        if not isinstance(scaler, (MinMaxScaler, StandardScaler)):
            raise ValueError("Provided scaler must be an instance of MinMaxScaler or StandardScaler.")

        data_test_scaled = scaler.transform(data)
        return data_test_scaled, scaler

    @staticmethod
    def inverse_scale(data: np.ndarray, scaler: MinMaxScaler | StandardScaler) -> np.ndarray:
        """
        Inversely scales the input data.

        Args:
            data: Scaled data to be inverse-transformed.
            scaler: Pre-trained scaler.

        Returns:
            Inverse-scaled data.

        """
        if not isinstance(scaler, (MinMaxScaler, StandardScaler)):
            raise ValueError("Provided scaler must be an instance of MinMaxScaler or StandardScaler.")

        data_inversed = scaler.inverse_transform(data)
        return data_inversed