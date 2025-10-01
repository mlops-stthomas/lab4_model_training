# Airflow Teaching Lab: Setup Guide

This repository contains a lightweight lab to help you set up Apache Airflow locally.  
Follow these instructions to configure your environment and start the Airflow web interface.

---

## Requirements
- Python 3.8+  
- Virtual environment (recommended)  
- AWS credentials (via IAM role or access keys)

---

## Setup Instructions

```bash
# Clone the repository

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Configure environment variables
source scripts/setup_env.sh

# Initialize the Airflow database
airflow db init

# Create an admin user for the Airflow UI
airflow users create \
  --username admin \
  --firstname Airflow \
  --lastname Admin \
  --role Admin \
  --email admin@example.com \
  --password admin

# Start Airflow services (use two separate terminals)

# Terminal 1: start the webserver
airflow webserver --port 8080

# Terminal 2: start the scheduler
airflow scheduler
