import os
import subprocess
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



def train_model(df: pd.DataFrame, model_path: str = "models/iris_model.pkl") -> float:
    """Train a logistic regression classifier and save it."""
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"[ml_pipeline.model] Model accuracy: {acc:.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"[ml_pipeline.model] Saved model to {model_path}")

    return acc

def evaluate_model(df: pd.DataFrame, model_path: str = "models/iris_model.pkl") -> float:

    # Loading Model from Stored Pickle File
    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")

    # Gathering X and y
    X = df.drop(columns=["target"])
    y = df["target"]

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    print(f"Evaluation Accuracy: {accuracy:.4f}")

    return accuracy

def promote_model(
    model_path: str = "models/iris_model.pkl",
    accuracy: float = 0.0,
    threshold: float = 0.95,
    bucket: str = "mlops-hw3-models-astar",
    s3_key: str = "models/iris_model.pkl"
) -> bool:

    print(f"Performing Check on {model_path}, which has an accuracy of {accuracy}, will upload to {bucket}/{s3_key} ")
    print(os.environ.values())
    """Promote model to S3 if accuracy meets the threshold."""
    if accuracy < threshold:
        print(f"Promotion skipped: {accuracy:.4f} < threshold {threshold:.4f}")
        return False
    print("Promotion Passed")


    # First tried boto3, but this cause deadlock issues, resorted to calling the aws cli directly
    result = subprocess.run(
        ["aws", "s3", "cp", model_path, f"s3://{bucket}/{s3_key}"],
        capture_output=True,
        text=True,
        timeout=30
    )

    if result.returncode != 0:
        raise RuntimeError(f"S3 upload failed: {result.stderr}")

    print(f"Promoted model to s3://{bucket}/{s3_key}")
    return True