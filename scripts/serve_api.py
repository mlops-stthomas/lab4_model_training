# scripts/serve_api.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from app.api import create_app
app = create_app("models/iris_model.pkl")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)