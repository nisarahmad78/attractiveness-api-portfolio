# utils/metrics.py
import numpy as np

class Metrics:
    """Utility class for evaluating model performance on regression tasks."""

    def __init__(self, preds, trues):
        """
        Initialize with predictions and true values.
        Args:
            preds (array-like): Model predictions
            trues (array-like): Ground truth values
        """
        self.preds = np.array(preds)
        self.trues = np.array(trues)

    def mae(self):
        """Mean Absolute Error"""
        return float(np.mean(np.abs(self.preds - self.trues)))

    def rmse(self):
        """Root Mean Squared Error"""
        return float(np.sqrt(np.mean((self.preds - self.trues) ** 2)))

    def accuracy_within(self, tol=0.5):
        """
        Accuracy within a given tolerance.
        Example: within ±0.5 of the true value.
        """
        return float(np.mean(np.abs(self.preds - self.trues) <= tol) * 100.0)

    def summary(self, tol=0.5):
        """Return all metrics in a single dictionary."""
        return {
            "MAE": self.mae(),
            "RMSE": self.rmse(),
            f"Accuracy_Within_{tol}": self.accuracy_within(tol)
        }
