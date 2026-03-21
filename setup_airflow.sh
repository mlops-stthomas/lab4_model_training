#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------
# Airflow Lab Setup Script (project-isolated)
# -------------------------------------------

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/venv"
AIRFLOW_HOME_DIR="$PROJECT_ROOT/airflow_home"

export AIRFLOW_HOME="$AIRFLOW_HOME_DIR"
export AIRFLOW__CORE__DAGS_FOLDER="$PROJECT_ROOT/dags"
export AIRFLOW__CORE__PLUGINS_FOLDER="$PROJECT_ROOT/plugins"
export AIRFLOW__LOGGING__BASE_LOG_FOLDER="$AIRFLOW_HOME/logs"
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="sqlite:///$AIRFLOW_HOME/airflow.db"
export AIRFLOW__CORE__LOAD_EXAMPLES="False"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "AIRFLOW_HOME=$AIRFLOW_HOME"

# Create local folders
mkdir -p "$AIRFLOW_HOME"
mkdir -p "$PROJECT_ROOT/dags"
mkdir -p "$PROJECT_ROOT/plugins"
mkdir -p "$AIRFLOW_HOME/logs"

# Activate local virtual environment if it exists
if [[ -d "$VENV_DIR" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  echo "Activated virtual environment: $VENV_DIR"
else
  echo "Warning: no virtual environment found at $VENV_DIR"
fi

# Allow callers (for example Docker containers) to source this script
# only for environment setup without reinitializing the metadata DB.
if [[ "${AIRFLOW_SKIP_DB_INIT:-0}" != "1" ]]; then
  # Initialize Airflow metadata DB locally
  airflow db init
else
  echo "Skipping airflow db init because AIRFLOW_SKIP_DB_INIT=1"
fi

echo
echo "Airflow lab environment is set up locally."
echo "Nothing was written to ~/.bashrc, ~/.zshrc, or ~/.profile."
echo
echo "To use this environment in the current shell:"
echo "  source ./setup_airflow.sh"
echo
echo "Then run:"
echo "  airflow webserver --port 8080"
echo "  airflow scheduler"