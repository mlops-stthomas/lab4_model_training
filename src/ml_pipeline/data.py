import os
import pandas as pd
from sklearn.datasets import load_iris

def generate_data(output_path: str = "data/iris.csv") -> str:
    """Generate (or download) dataset and save as CSV."""
    iris = load_iris(as_frame=True)
    df = iris.frame
    df["target"] = iris.target

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[ml_pipeline.data] Saved dataset to {output_path}")
    return output_path

def load_data(data_path: str = "data/iris.csv") -> pd.DataFrame:
    """Load dataset from CSV into a dataframe."""
    return pd.read_csv(data_path)