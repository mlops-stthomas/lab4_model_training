# Copyright (c) 2026 Luca and Pam. All rights reserved.

import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

BUCKET_NAME = "lab4-model-training"
MODEL_PREFIX = "models/"


class S3:
    def __init__(
        self,
        bucket_name: str = BUCKET_NAME,
        endpoint_url: str | None = None,
    ):
        self.bucket_name = bucket_name
        self.endpoint_url = endpoint_url or os.getenv("S3_ENDPOINT_URL", "http://localstack:4566")
        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        )
        self.ensure_bucket()

    def ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
        except ClientError:
            self.client.create_bucket(Bucket=self.bucket_name)

    def upload_file(self, file_path: str, object_name: str) -> None:
        source = Path(file_path)
        if not source.exists():
            raise FileNotFoundError(f"File not found: {source}")

        print(f"Uploading {source} to s3://{self.bucket_name}/{object_name}...")
        self.client.upload_file(str(source), self.bucket_name, object_name)

    def upload_model_artifacts(self, model_name: str, version: str | None = None) -> None:
        """
        Upload model artifacts to S3, optionally to a versioned directory.
        
        Args:
            model_name: The name of the model (e.g., 'iris', 'cancer')
            version: Optional version string (datetime). If provided, uploads to models/<model_name>/<version>/
                    If not provided, uploads to models/ (legacy behavior)
        """
        if version is not None:
            # Upload to versioned directory
            version_prefix = f"{MODEL_PREFIX}{model_name}/{version}/"
            local_dir = Path(f"models/{model_name}/{version}")
            
            files = {
                f"{version_prefix}{model_name}.pkl": f"models/{model_name}/{version}/{model_name}.pkl",
                f"{version_prefix}metrics.json": f"models/{model_name}/{version}/metrics.json",
                f"{version_prefix}metadata.json": f"models/{model_name}/{version}/metadata.json",
            }
            
            # Upload versioned artifacts
            for object_name, local_path in files.items():
                self.upload_file(local_path, object_name)
            
            # Also upload the promoted_model.json to point to this version
            promoted_model_local = f"models/{model_name}/promoted_model.json"
            promoted_model_s3 = f"{MODEL_PREFIX}{model_name}/promoted_model.json"
            if Path(promoted_model_local).exists():
                self.upload_file(promoted_model_local, promoted_model_s3)
        else:
            # Legacy non-versioned upload
            files = {
                f"{MODEL_PREFIX}{model_name}.pkl": f"models/{model_name}.pkl",
                f"{MODEL_PREFIX}metrics.json": "models/metrics.json",
                f"{MODEL_PREFIX}metadata.json": "models/metadata.json",
            }

            for object_name, local_path in files.items():
                self.upload_file(local_path, object_name)

