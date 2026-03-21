# Copyright (c) 2026 Luca and Pam. All rights reserved.

import json
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from model.base_model import BaseModel
from utils.s3 import S3


class IrisModel(BaseModel):
    def __init__(self):
        super().__init__(name="iris")
        self.dataset = "iris"
        self.model_type = "logistic_regression"

    def train(self, model_path: str) -> None:
        data = load_iris()
        X, y = data.data, data.target
        print(f"Training Iris model with {X.shape[0]} samples and {X.shape[1]} features...")

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=1 - self.split_ratio, random_state=self.random_state
        )

        model = LogisticRegression(max_iter=300)
        model.fit(X_train, y_train)

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        print("Iris model training completed.")

    def evaluate(self, model_path: str) -> dict:
        print("Evaluating Iris model...")
        model = joblib.load(model_path)

        data = load_iris()
        X, y = data.data, data.target
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=1 - self.split_ratio, random_state=self.random_state
        )

        y_pred = model.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))
        metrics = {"model_name": self.name, "accuracy": accuracy}

        Path("models").mkdir(parents=True, exist_ok=True)
        with open("models/metrics.json", "w") as f:
            json.dump(metrics, f)

        with open("models/metadata.json", "w") as f:
            json.dump(self.generate_metadata(accuracy), f)

        print(f"Iris model accuracy: {accuracy:.4f}")
        return metrics

    def promote(self, metrics: dict) -> None:
        print("Promoting Iris model...")
        versioned_dir = self.get_versioned_model_dir()
        versioned_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all model files to the versioned directory
        import shutil
        shutil.copy("models/iris_model.pkl", versioned_dir / "iris_model.pkl")
        shutil.copy("models/metadata.json", versioned_dir / "metadata.json")
        shutil.copy("models/metrics.json", versioned_dir / "metrics.json")
        
        # Save promoted model info at the root level pointing to this version
        Path(f"models/{self.name}").mkdir(parents=True, exist_ok=True)
        with open(f"models/{self.name}/promoted_model.json", "w") as f:
            json.dump(
                {
                    "model_name": self.name,
                    "version": self.version,
                    "metrics": metrics,
                    "timestamp": self.version
                },
                f
            )
        
        print(f"Iris model promoted to version {self.version}")
        s3 = S3()
        s3.upload_model_artifacts(self.name, self.version)