import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class CustomerLTVRegressor:
    """
    Supervised Gradient Boosting Regressor model to predict 12-month Customer Lifetime Value (LTV).
    """
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.05, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state
        )
        self.feature_names = []
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list = None) -> dict:
        """
        Trains the Gradient Boosting Regressor on an 80/20 train/test split.
        """
        if feature_names:
            self.feature_names = feature_names

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=self.random_state
        )

        self.model.fit(X_train, y_train)
        self.fitted = True

        y_pred = self.model.predict(X_test)

        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        importances = {}
        if self.feature_names and hasattr(self.model, 'feature_importances_'):
            for name, imp in zip(self.feature_names, self.model.feature_importances_):
                importances[name] = float(round(imp, 4))

        return {
            'r2_score': round(r2, 4),
            'mean_absolute_error': round(mae, 2),
            'root_mean_squared_error': round(rmse, 2),
            'feature_importances': importances
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts 12-month Customer LTV ($) for input matrix X.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before predicting.")
        return np.maximum(0.0, self.model.predict(X))

    def save(self, filepath: str):
        """
        Saves the trained LTV regressor model artifact to disk.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "CustomerLTVRegressor":
        """
        Loads trained LTV regressor model artifact from disk.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"LTV regressor model file not found: {filepath}")
        return joblib.load(filepath)
