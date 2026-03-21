# Copyright (c) 2026 Luca and Pam. All rights reserved.

from enum import Enum
from typing import Type

from model.base_model import BaseModel
from model.cancer import CancerModel
from model.iris import IrisModel


class Model(Enum):
    IRIS = "iris"
    CANCER = "cancer"


class ModelFactory:
    @staticmethod
    def get_model_class(model: Model) -> Type[BaseModel]:
        if model == Model.IRIS:
            return IrisModel
        if model == Model.CANCER:
            return CancerModel
        raise ValueError(f"Unsupported model: {model}")

    @staticmethod
    def create_model(model: Model) -> BaseModel:
        return ModelFactory.get_model_class(model)()

