#!/usr/bin/env bash
set -e

# -------------------------------------------
# Airflow Lab Setup Script
# -------------------------------------------

AIRFLOW_HOME_DIR="$(pwd)/airflow_home"
SHELL_RC=""

# Detect which shell rc file to use
if [[ -n "$ZSH_VERSION" ]]; then
  SHELL_RC="$HOME/.zshrc"
elif [[ -n "$BASH_VERSION" ]]; then
  SHELL_RC="$HOME/.bashrc"
else
  SHELL_RC="$HOME/.profile"
fi

# 1. Persist AIRFLOW_HOME in shell rc
if ! grep -q "AIRFLOW_HOME=" "$SHELL_RC"; then
  echo "export AIRFLOW_HOME=$AIRFLOW_HOME_DIR" >> "$SHELL_RC"
  echo "Added AIRFLOW_HOME to $SHELL_RC"
else
  echo "AIRFLOW_HOME already set in $SHELL_RC"
fi

# Also set for this session so script works immediately
export AIRFLOW_HOME=$AIRFLOW_HOME_DIR
mkdir -p "$AIRFLOW_HOME"

# 2. Initialize Airflow metadata DB
echo "Initializing Airflow database..."
airflow db init

# 3. Create admin user (skip if exists)
echo "Creating admin user (username=admin, password=admin)..."
airflow users create \
  --username admin \
  --password admin \
  --firstname Air \
  --lastname Flow \
  --role Admin \
  --email admin@example.com || true

# 4. Symlink project DAGs into AIRFLOW_HOME
if [ ! -L "$AIRFLOW_HOME/dags" ]; then
  echo "Linking project dags/ into $AIRFLOW_HOME/dags..."
  rm -rf "$AIRFLOW_HOME/dags"
  ln -s "$(pwd)/dags" "$AIRFLOW_HOME/dags"
else
  echo "DAGs already linked."
fi

echo ""
echo "✅ Airflow setup complete!"
echo "Open a new terminal or run: source $SHELL_RC"
echo ""
echo "Then start Airflow:"
echo "  Terminal 1: airflow scheduler"
echo "  Terminal 2: airflow webserver --port 8080"
echo "Login at http://localhost:8080 with admin / admin"
