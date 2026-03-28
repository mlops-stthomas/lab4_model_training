# Lab 4: Model Training and Serving with Airflow + FastAPI

In this lab you will build an **end-to-end ML pipeline** using Apache Airflow and serve the trained model with FastAPI.  

The pipeline includes:  
1. **Generate Data** – downloads the Iris dataset and saves it as a CSV.  
2. **Train Model** – trains a Logistic Regression classifier.  
3. **Pipeline** – runs both steps end-to-end.  
4. **Serve Model** – starts a FastAPI app for inference.  

---

## 📂 Project Structure

```
lab4_model_training/
├── dags/                        # Airflow DAGs
│   ├── ml_pipeline_dag.py       # full pipeline: generate + train
│   ├── generate_data_dag.py     # generate dataset only
│   └── train_model_dag.py       # train model only
├── src/
│   ├── ml_pipeline/             # training pipeline
│   │   ├── data.py
│   │   └── model.py
│   └── app/                     # serving app
│       └── api.py
├── scripts/
│   ├── generate_data.py         # CLI wrapper
│   ├── train_model.py           # CLI wrapper
│   └── serve_api.py             # runs FastAPI app
├── data/                        # dataset outputs
│   └── iris.csv
├── models/                      # trained models
│   └── iris_model.pkl
├── airflow_home/                # Airflow metadata (created after setup)
├── requirements.txt             # Python dependencies
└── setup_airflow.sh             # one-time setup script
```

---

## 🛠 Environment Setup

We use **one virtual environment** for all labs.

1. Create and activate:

```
python3 -m venv ~/venvs/airflow-class
source ~/venvs/airflow-class/bin/activate
```

2. Install dependencies:

```
pip install -r requirements.txt
```

⚠️ The `requirements.txt` pins **Airflow 2.10.2**. If you are not on Python 3.10, update the constraints line to match (`constraints-3.9.txt` or `constraints-3.11.txt`).  

---

## ⚙️ Airflow Setup (one time)

Run the setup script:

```
./setup_airflow.sh
```

This will:  
- Set `AIRFLOW_HOME` inside this project.  
- Initialize the Airflow database.  
- Create an admin user (`admin / admin`).  
- Symlink your `dags/` folder into Airflow’s DAGs directory.  

Afterwards, open a new terminal (or `source ~/.bashrc` / `~/.zshrc`) so `$AIRFLOW_HOME` is available automatically.  

---

## 🐳 Running Airflow with Docker Compose (recommended)

This repo now includes a Docker Compose stack that runs:
- Airflow scheduler
- Airflow webserver

`airflow-webserver` sources `setup_airflow.sh` on startup and performs one-time initialization.

Start the stack:

```bash
python scripts/control_stack.py --up --detach
```

Check status:

```bash
python scripts/control_stack.py --status
```

Show logs:

```bash
python scripts/control_stack.py --logs --logs-tail 200
```

Stop stack:

```bash
python scripts/control_stack.py --stop
```

Bring down stack:

```bash
python scripts/control_stack.py --down
```

Visit Airflow UI 👉 http://localhost:8080  
Login: `admin / admin`

## 🚀 Running Airflow (local, non-Docker)

Use two terminals:

**Terminal 1 – Scheduler**
```
source ~/venvs/airflow-class/bin/activate
airflow scheduler
```

**Terminal 2 – Webserver**
```
source ~/venvs/airflow-class/bin/activate
airflow webserver --port 8080 --host 0.0.0.0
```

Then visit 👉 http://<ipaddress>:8080  
Login: `admin / admin`

Replace `<ipaddress>` with your EC2 instance’s **public IPv4 address**.  

---

## 📊 DAGs to Explore

You will see three DAGs:

1. **`generate_data_only`**  
   - Saves `data/iris.csv`.

2. **`train_model_only`**  
   - Trains a Logistic Regression model from CSV.  
   - Produces `models/iris_model.pkl`.

3. **`ml_pipeline`**  
   - End-to-end pipeline:  
     `generate_data` → `train_model`.

---

## 🧪 Testing Without Airflow

You can also run scripts directly:

```
python scripts/generate_data.py
python scripts/train_model.py
```

This will produce `data/iris.csv` and `models/iris_model.pkl`.  

---

## 🌐 Serving the Model with FastAPI

After training the model, you can serve it with FastAPI.

1. Run the API:

```
python scripts/serve_api.py
```

Or start it from the control CLI:

```bash
python scripts/control_stack.py --start-api
```

You can also pass host/port/model path:

```bash
python scripts/control_stack.py --start-api --api-host 0.0.0.0 --api-port 8000 --api-model-path models/iris_model.pkl
```

2. Open docs: http://<ipaddress>:8000/docs  

3. Try a prediction in Swagger UI with four required features:

```
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Response:

```
{"prediction": "setosa", "class_index": 0}
```

---

## 🌸 Example Inputs

Values that commonly predict each class:

- **Setosa**: `5.1, 3.5, 1.4, 0.2`  
- **Versicolor**: `6.0, 2.9, 4.5, 1.5`  
- **Virginica**: `6.9, 3.1, 5.4, 2.1`  

---

## ✅ Summary

By the end of this lab you will have:  
- Built a training pipeline with Airflow.  
- Produced a dataset and a trained model artifact.  
- Served the trained model with FastAPI.  
- Sent live inference requests to your model.  

Next steps: containerize this API and deploy it to the cloud 🚀
