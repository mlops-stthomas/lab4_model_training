from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys, os

# Ensure src/ is on path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from ml_pipeline.data import load_data
from ml_pipeline.model import train_model

default_args = {"owner": "airflow", "retries": 1}

with DAG(
    dag_id="train_model_only",
    default_args=default_args,
    description="Train ML model only (expects data to already exist)",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    def train_model_wrapper(data_path: str, model_path: str):
        df = load_data(data_path)
        return train_model(df, model_path)

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model_wrapper,
        op_kwargs={
            "data_path": "data/iris.csv",
            "model_path": "models/iris_model.pkl",
        },
    )