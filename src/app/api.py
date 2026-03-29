# src/app/api.py
import joblib
import json
import subprocess
import tempfile
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class BreastCancerRequest(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float

    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float

    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float


FEATURE_ORDER = [
    "mean_radius", "mean_texture", "mean_perimeter", "mean_area",
    "mean_smoothness", "mean_compactness", "mean_concavity", "mean_concave_points",
    "mean_symmetry", "mean_fractal_dimension",
    "radius_error", "texture_error", "perimeter_error", "area_error",
    "smoothness_error", "compactness_error", "concavity_error", "concave_points_error",
    "symmetry_error", "fractal_dimension_error",
    "worst_radius", "worst_texture", "worst_perimeter", "worst_area",
    "worst_smoothness", "worst_compactness", "worst_concavity", "worst_concave_points",
    "worst_symmetry", "worst_fractal_dimension",
]


def load_latest_model_from_s3(bucket: str, prefix: str = "models/breast_cancer/"):
    """
    Lists all versioned folders under `prefix`, picks the lexicographically
    latest one (versions are YYYYmmddHHMMSS so this is also chronologically
    latest), then downloads model.pkl and metadata.json from that folder via
    the aws cli.

    Returns:
        model:    the loaded sklearn model
        metadata: dict with model_version, dataset, model_type, accuracy
    """
    s3_prefix_uri = f"s3://{bucket}/{prefix}"

    # List versioned subfolders
    result = subprocess.run(
        ["aws", "s3api", "list-objects-v2",
         "--bucket", bucket,
         "--prefix", prefix,
         "--delimiter", "/",
         "--output", "json"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to list S3 versions: {result.stderr}")

    response = json.loads(result.stdout)
    common_prefixes = [cp["Prefix"] for cp in response.get("CommonPrefixes", [])]

    if not common_prefixes:
        raise RuntimeError(
            f"No model versions found in {s3_prefix_uri}. "
            "Train and promote a model first."
        )

    latest_prefix = sorted(common_prefixes)[-1]
    print(f"[api] Latest model version prefix: {latest_prefix}")

    # Download model.pkl into a temp file then load it
    model_key = f"{latest_prefix}model.pkl"
    with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp:
        result = subprocess.run(
            ["aws", "s3", "cp", f"s3://{bucket}/{model_key}", tmp.name],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to download model from S3: {result.stderr}")
        model = joblib.load(tmp.name)

    # Stream metadata.json directly to stdout using "-" as the destination
    metadata_key = f"{latest_prefix}metadata.json"
    result = subprocess.run(
        ["aws", "s3", "cp", f"s3://{bucket}/{metadata_key}", "-"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to download metadata from S3: {result.stderr}")

    metadata = json.loads(result.stdout)
    print(f"[api] Model loaded successfully — version: {metadata['model_version']}")

    return model, metadata


def create_app(
    bucket: str = "mlops-hw3-models-astar",
    s3_prefix: str = "models/breast_cancer/",
):
    """
    Creates a FastAPI app that serves breast cancer predictions using the
    latest promoted model version from S3.
    """
    model, metadata = load_latest_model_from_s3(bucket, s3_prefix)

    app = FastAPI(title="Breast Cancer Model API")

    target_names = {0: "malignant", 1: "benign"}

    @app.get("/")
    def root():
        return {
            "message": "Breast Cancer model is ready for inference!",
            "model_version": metadata["model_version"],
            "classes": target_names,
        }

    @app.get("/model/info")
    def model_info():
        return {
            "model_version": metadata["model_version"],
            "dataset":       metadata["dataset"],
            "model_type":    metadata["model_type"],
            "accuracy":      metadata["accuracy"],
        }

    @app.post("/predict")
    def predict(request: BreastCancerRequest):
        X = np.array([[getattr(request, f) for f in FEATURE_ORDER]])
        try:
            idx = int(model.predict(X)[0])
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {
            "prediction":    target_names[idx],
            "class_index":   idx,
            "model_version": metadata["model_version"],
        }

    return app