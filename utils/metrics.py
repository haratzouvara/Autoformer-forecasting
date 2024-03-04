import numpy as np


class Metrics:
    """
    A class for calculating evaluation metrics between actual and predicted values.

    Args:
      actual: The actual values.
      predicted: The predicted values.

    Methods:
      rmse(): Calculate the Root Mean Squared Error (RMSE) between actual and predicted values.

    Raises:
        ValueError: If the shapes of actual values and predicted values are not the same.

    """

    def __init__(self, actual: np.ndarray, predicted: np.ndarray):

        self.actual = actual
        self.predicted = predicted

    def rmse(self) -> float:
        """
        Calculate the Root Mean Squared Error (RMSE) between actual and predicted values.

        Returns:
        - float: The RMSE score.
        """

        if self.actual.shape != self.predicted.shape:
            raise ValueError("Shape of actual values and predicted values must be the same")

        rmse_value = np.sqrt(np.mean((self.actual - self.predicted) ** 2))

        return rmse_value
