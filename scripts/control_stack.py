#!/usr/bin/env python3
# Copyright (c) 2026 Luca and Pam. All rights reserved.

"""Flag-based CLI to control Docker Compose Airflow stack and local API server."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"
SERVE_API_SCRIPT = PROJECT_ROOT / "scripts" / "serve_api.py"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "iris_model.pkl"


def run_command(command: list[str], check: bool = True) -> int:
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"$ {printable}")
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if check and result.returncode != 0:
        raise SystemExit(result.returncode)
    return result.returncode


def compose_base_command() -> list[str]:
    return ["docker", "compose", "-f", str(COMPOSE_FILE)]


def start_compose(detach: bool) -> None:
    command = compose_base_command() + ["up"]
    if detach:
        command.append("-d")
    run_command(command)


def stop_compose() -> None:
    run_command(compose_base_command() + ["stop"])


def down_compose(remove_volumes: bool) -> None:
    command = compose_base_command() + ["down"]
    if remove_volumes:
        command.append("-v")
    run_command(command)


def restart_compose(detach: bool) -> None:
    stop_compose()
    start_compose(detach=detach)


def status_compose() -> None:
    run_command(compose_base_command() + ["ps"])


def logs_compose(follow: bool, tail: int) -> None:
    command = compose_base_command() + ["logs", "--tail", str(tail)]
    if follow:
        command.append("-f")
    run_command(command)


def start_api(host: str, port: int, model_path: str) -> None:
    resolved_model_path = Path(model_path)
    if not resolved_model_path.is_absolute():
        resolved_model_path = (PROJECT_ROOT / resolved_model_path).resolve()

    if not resolved_model_path.exists():
        print(
            "Error: model file was not found at "
            f"{resolved_model_path}. Train the model first using Airflow or scripts/train_model.py.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    command = [
        sys.executable,
        str(SERVE_API_SCRIPT),
        "--host",
        host,
        "--port",
        str(port),
        "--model-path",
        str(resolved_model_path),
    ]
    run_command(command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Control Airflow Docker Compose stack and start local FastAPI server."
    )
    parser.add_argument("--up", action="store_true", help="Start Docker Compose services.")
    parser.add_argument("--stop", action="store_true", help="Stop running Compose services.")
    parser.add_argument("--down", action="store_true", help="Remove Compose services.")
    parser.add_argument("--restart", action="store_true", help="Restart Compose services.")
    parser.add_argument("--status", action="store_true", help="Show Compose service status.")
    parser.add_argument("--logs", action="store_true", help="Show Compose logs.")
    parser.add_argument(
        "--logs-follow",
        action="store_true",
        help="Follow logs stream (used with --logs).",
    )
    parser.add_argument(
        "--logs-tail",
        type=int,
        default=100,
        help="Number of log lines to show (used with --logs).",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Run docker compose up in detached mode (used with --up/--restart).",
    )
    parser.add_argument(
        "--remove-volumes",
        action="store_true",
        help="Also remove volumes when used with --down.",
    )
    parser.add_argument(
        "--start-api",
        action="store_true",
        help="Start local FastAPI server via scripts/serve_api.py.",
    )
    parser.add_argument("--api-host", default="0.0.0.0", help="API host binding.")
    parser.add_argument("--api-port", type=int, default=8000, help="API port.")
    parser.add_argument(
        "--api-model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to trained model file.",
    )

    args = parser.parse_args()

    operations_selected = any(
        [args.up, args.stop, args.down, args.restart, args.status, args.logs, args.start_api]
    )
    if not operations_selected:
        parser.error("No operation selected. Use one or more of: --up --stop --down --restart --status --logs --start-api")

    if args.logs_follow and not args.logs:
        parser.error("--logs-follow requires --logs")

    return args


def main() -> None:
    args = parse_args()

    if args.stop:
        stop_compose()
    if args.down:
        down_compose(remove_volumes=args.remove_volumes)
    if args.restart:
        restart_compose(detach=args.detach)
    if args.up:
        start_compose(detach=args.detach)
    if args.status:
        status_compose()
    if args.logs:
        logs_compose(follow=args.logs_follow, tail=args.logs_tail)
    if args.start_api:
        start_api(host=args.api_host, port=args.api_port, model_path=args.api_model_path)


if __name__ == "__main__":
    main()
