import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

class SegmentClassifierModel:
    """
    Supervised Random Forest Classifier trained on clustering assignments
    to score new customer profiles in real time with probabilistic confidence.
    """
    def __init__(self, n_estimators: int = 100, max_depth: int = 12, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the Random Forest Classifier on feature matrix X and target labels y.
        """
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts segment label for input matrix X.
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns prediction probability distribution across all segments.
        """
        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluates classifier performance on test set.
        """
        y_pred = self.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="weighted"))
        report = classification_report(y_test, y_pred, output_dict=True)
        return {
            "accuracy": round(acc, 4),
            "weighted_f1_score": round(f1, 4),
            "report": report
        }

    def get_feature_importances(self, feature_names: list[str]) -> list[dict]:
        """
        Returns feature importances ranked by predictive power.
        """
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        return [
            {
                "feature": feature_names[i],
                "importance": round(float(importances[i]), 4)
            }
            for i in indices
        ]

    def save(self, file_path: str):
        """
        Saves fitted model artifact to disk.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path: str):
        """
        Loads fitted model artifact from disk.
        """
        return joblib.load(file_path)
