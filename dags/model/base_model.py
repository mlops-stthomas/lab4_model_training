# Copyright (c) 2026 Luca and Pam. All rights reserved.

from __future__ import annotations

from datetime import datetime
from pathlib import Path


class BaseModel:
    def __init__(self, name: str):
        self.name = name
        self.split_ratio = 0.8
        self.random_state = 42
        self.version = self.generate_version()
        self.dataset = ""
        self.model_type = ""
        self.threshold = 0.94

    def generate_version(self) -> str:
        return datetime.now().strftime("%Y%m%d%H%M%S")

    def get_versioned_model_dir(self) -> Path:
        """Get the versioned directory path using the datetime version."""
        model_dir = Path(f"models/{self.name}/{self.version}")
        return model_dir

    def generate_metadata(self, accuracy: float) -> dict:
        return {
            "model_name": self.name,
            "model_version": self.version,
            "dataset": self.dataset,
            "model_type": self.model_type,
            "accuracy": accuracy,
        }

    def train(self, model_path: str) -> None:
        raise NotImplementedError

    def evaluate(self, model_path: str) -> dict:
        raise NotImplementedError

    def promote(self, metrics: dict) -> None:
        raise NotImplementedError
