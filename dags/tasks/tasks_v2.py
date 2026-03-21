# Copyright (c) 2026 Luca and Pam. All rights reserved.

from airflow.decorators import task

from tasks.model_factory import Model, ModelFactory


@task(task_id="train_model_v2")
def train(model_name: str) -> dict:
    model_class = ModelFactory.get_model_class(Model(model_name.lower()))
    model = model_class()
    model_path = f"models/{model.name}.pkl"
    print(f"Training {model.name} model...")
    model.train(model_path)
    return {"model_name": model.name, "model_path": model_path}


@task(task_id="evaluate_model_v2")
def evaluate(train_result: dict) -> dict:
    model_class = ModelFactory.get_model_class(Model(train_result["model_name"]))
    model = model_class()
    print(f"Evaluating {model.name} model...")
    metrics = model.evaluate(train_result["model_path"])
    return {"model_name": model.name, "metrics": metrics}


@task(task_id="promote_model_v2")
def promote(eval_result: dict) -> bool:
    model_class = ModelFactory.get_model_class(Model(eval_result["model_name"]))
    model = model_class()
    print(f"Promoting {model.name} model...")
    model.promote(eval_result["metrics"])
    return True
