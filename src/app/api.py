# src/app/api.py
import json
import os
import joblib
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
from botocore.exceptions import ClientError

# Explicit request schema for Iris dataset (4 features)
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


def get_s3_client():
    """Create and return an S3 client configured for the environment."""
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL", "http://localstack:4566"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )


def load_metadata_from_s3(bucket_name: str = "lab4-model-training", model_name: str = "cancer"):
    """Load model metadata from S3, getting the latest version."""
    s3_client = get_s3_client()
    try:
        # First, try to get the promoted_model.json to find the current version
        promoted_response = s3_client.get_object(Bucket=bucket_name, Key=f"models/{model_name}/promoted_model.json")
        promoted_info = json.loads(promoted_response["Body"].read())
        version = promoted_info.get("version")
        
        # Load metadata from the versioned directory
        metadata_key = f"models/{model_name}/{version}/metadata.json"
        response = s3_client.get_object(Bucket=bucket_name, Key=metadata_key)
        metadata = json.loads(response["Body"].read())
        return metadata
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "NoSuchKey":
            raise HTTPException(
                status_code=404,
                detail=f"No promoted model found for '{model_name}'. Run the training pipeline first to create and promote a model."
            )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load metadata from S3: {str(e)}"
        )

def create_app(model_path: str = "models/iris_model.pkl"):
    """
    Creates a FastAPI app that serves predictions for the Iris model.

    Example values that commonly predict each class:
      - setosa:     5.1, 3.5, 1.4, 0.2
      - versicolor: 6.0, 2.9, 4.5, 1.5
      - virginica:  6.9, 3.1, 5.4, 2.1
    """
    # Helpful guard so students get a clear error if they forgot to train first
    # if not Path(model_path).exists():
    #     raise RuntimeError(
    #         f"Model file not found at '{model_path}'. "
    #         "Train the model first (run the DAG or scripts/train_model.py)."
    #     )

    if Path(model_path).exists():
        model = joblib.load(model_path)
    app = FastAPI(title="Iris Model API")

    # Map numeric predictions to class names
    target_names = {0: "setosa", 1: "versicolor", 2: "virginica"}

    @app.get("/")
    def root():
        return {
            "message": "Iris model is ready for inference!",
            "classes": target_names,
        }

    @app.post("/predict")
    def predict(request: IrisRequest):
        # Convert request into the correct shape (1 x 4)
        X = np.array([
            [request.sepal_length, request.sepal_width,
             request.petal_length, request.petal_width]
        ])
        try:
            idx = int(model.predict(X)[0])
        except Exception as e:
            # Surface any shape/validation issues as a 400 instead of a 500
            raise HTTPException(status_code=400, detail=str(e))
        return {"prediction": target_names[idx], "class_index": idx}
    
    @app.get("/model/info")
    def get_model_info(model_name: str = "cancer"):
        metadata = load_metadata_from_s3(model_name=model_name)
        return metadata

    # return the FastAPI app
    return app
