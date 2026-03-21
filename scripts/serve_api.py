import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the Iris FastAPI model API.")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface to bind the API server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the API server.",
    )
    parser.add_argument(
        "--model-path",
        default="models/iris_model.pkl",
        help="Path to the trained model (relative to project root or absolute).",
    )
    return parser.parse_args()


def resolve_model_path(model_path: str) -> str:
    candidate = Path(model_path)
    if candidate.is_absolute():
        return str(candidate)
    return str((PROJECT_ROOT / candidate).resolve())


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_path)

    from app.api import create_app

    app = create_app(model_path)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()