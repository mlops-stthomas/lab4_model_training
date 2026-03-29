from airflow import DAG
from airflow.operators.python import PythonOperator
import sys, os
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from ml_pipeline.data import generate_data, load_data
from ml_pipeline.model import train_model, evaluate_model, promote_model

default_args = {
    "owner": "airflow",
    "retries": 1,
    "promotion_threshold": 0.95,
}

with DAG(
    dag_id="ml_training_pipeline_v2",
    default_args=default_args,
    description="Pipeline: Trains Model -> Evaluates model -> Promotes model if performance threshold met",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    # Train Model Step
    def train_model_wrapper(data_path: str, model_path_template: str, **context):
        run_id = context["run_id"].replace(":", "_").replace("+", "_")
        model_path = model_path_template.format(run_id=run_id)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        df = load_data(data_path)
        train_model(df, model_path)
        context["ti"].xcom_push(key="model_path", value=model_path)

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model_wrapper,
        retries=1,
        op_kwargs={
            "data_path": "data/breast_cancer.csv",
            "model_path_template": "models/{run_id}/breast_cancer_model.pkl",
        },
    )

    # Evaluate Model Step
    def evaluate_model_wrapper(data_path: str, **context):
        model_path = context["ti"].xcom_pull(task_ids="train_model", key="model_path")
        df = load_data(data_path)
        acc = evaluate_model(df, model_path)
        context["ti"].xcom_push(key="accuracy", value=acc)

    evaluate_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model_wrapper,
        retries=1,
        op_kwargs={
            "data_path": "data/breast_cancer.csv",
        },
    )

    # Promote Model Step
    def promote_model_wrapper(**context):
        model_path = context["ti"].xcom_pull(task_ids="train_model", key="model_path")
        accuracy = context["ti"].xcom_pull(task_ids="evaluate_model", key="accuracy")
        threshold = context["dag"].default_args.get("promotion_threshold", 0.95)

        timestamp = context.get("execution_date")
        if timestamp:
            model_version = timestamp.strftime("%Y%m%d%H%M%S")
        else:
            model_version = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d%H%M%S")

        promote_model(
            model_path=model_path,
            accuracy=accuracy,
            threshold=threshold,
            bucket="mlops-hw3-models-astar",
            model_version=model_version,
        )

    promote_task = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model_wrapper,
        retries=1,
    )

    train_task >> evaluate_task >> promote_task