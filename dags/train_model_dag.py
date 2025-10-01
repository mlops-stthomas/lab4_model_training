from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "../scripts"))
from train_model import train_and_save_model

default_args = {"owner": "airflow", "retries": 1}

with DAG(
    dag_id="train_model_only",
    default_args=default_args,
    description="Train ML model only (expects data to already exist)",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_and_save_model,
        op_kwargs={
            "data_path": "data/iris.csv",
            "model_path": "models/iris_model.pkl",
        },
    )