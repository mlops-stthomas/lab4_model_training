# Copyright (c) 2026 Luca and Pam. All rights reserved.

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param

from tasks.model_factory import Model
from tasks.tasks_v2 import evaluate, promote, train


default_args = {"owner": "airflow", "retries": 1}

with DAG(
    dag_id="ml_training_pipeline_v2",
    default_args=default_args,
    description="Train, evaluate, and promote ML model",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    params={
        "model_name": Param(
            Model.IRIS.value,
            enum=[model.value for model in Model],
            type="string",
        ),
    },
) as dag:
    # Runtime-selected model from Trigger DAG params.
    selected_model = "{{ params.model_name }}"

    train_result = train(model_name=selected_model)
    eval_result = evaluate(train_result=train_result)
    promote(eval_result=eval_result)
