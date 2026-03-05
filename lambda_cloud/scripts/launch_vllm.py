#!/usr/bin/env python3
"""Manage vLLM on a Lambda Cloud instance via SSH.

Usage:
    # Check status
    uv run python -m lambda_cloud.scripts.launch_vllm --ip 1.2.3.4 --status

    # Stop running server
    uv run python -m lambda_cloud.scripts.launch_vllm --ip 1.2.3.4 --stop

    # Start a model
    uv run python -m lambda_cloud.scripts.launch_vllm --ip 1.2.3.4 --model meta-llama/Llama-3.1-8B-Instruct

    # Start with SSH tunnel
    uv run python -m lambda_cloud.scripts.launch_vllm --ip 1.2.3.4 --model meta-llama/Llama-3.1-8B-Instruct --tunnel
"""

import argparse
import logging
import os
import signal
import sys

import yaml

from lambda_cloud.ssh import SSHConnection
from lambda_cloud.vllm_server import install_vllm, start_vllm, wait_for_vllm_ready, stop_vllm, vllm_status

DEFAULT_CONFIG = "lambda_cloud/config/lambda.yaml"


def main():
    parser = argparse.ArgumentParser(description="Manage vLLM on a Lambda Cloud instance")
    parser.add_argument("--ip", required=True, help="Instance IP address")
    parser.add_argument("--key-file", default="~/.ssh/anusha-cre-lambda-key.pem", help="SSH key file")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")

    # Actions (mutually exclusive)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--status", action="store_true", help="Check if vLLM is running and what model is served")
    action.add_argument("--stop", action="store_true", help="Stop the running vLLM server")

    # Launch options
    parser.add_argument("--model", help="HuggingFace model ID to serve")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to lambda.yaml config")
    parser.add_argument("--venv-path", default="/home/ubuntu/vllm-venv", help="Virtualenv path on remote")
    parser.add_argument("--extra-args", default=None, help="Extra vLLM serve arguments (overrides config)")
    parser.add_argument("--tunnel", action="store_true", help="Open SSH tunnel after starting")
    parser.add_argument("--local-port", type=int, default=8000, help="Local port for SSH tunnel")
    parser.add_argument("--timeout", type=int, default=900, help="Readiness timeout in seconds")
    parser.add_argument("--skip-install", action="store_true", help="Skip vLLM installation")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    ssh = SSHConnection(ip=args.ip, key_file=args.key_file)

    # ── Status ──
    if args.status:
        info = vllm_status(ssh, port=args.port)
        if info:
            model = info["model"] or "(unknown)"
            print(f"vLLM is running on {args.ip}")
            print(f"  Model:   {model}")
            print(f"  PID:     {info['pid']}")
            print(f"  Command: {info['cmdline']}")
        else:
            print(f"vLLM is not running on {args.ip}")
        return

    # ── Stop ──
    if args.stop:
        info = vllm_status(ssh, port=args.port)
        if info:
            model = info["model"] or "(unknown)"
            print(f"Stopping vLLM on {args.ip} (model={model}, pid={info['pid']})...")
            stop_vllm(ssh)
            print("Stopped.")
        else:
            print(f"vLLM is not running on {args.ip}")
        return

    # ── Launch ──
    if not args.model:
        parser.error("--model is required when launching (or use --status/--stop)")

    # Load vllm_args from config if not overridden via CLI
    extra_args = args.extra_args
    if extra_args is None:
        try:
            with open(args.config) as f:
                config = yaml.safe_load(f)
            gpu_map = config.get("model_gpu_map", {})
            model_cfg = gpu_map.get(args.model) or gpu_map.get("_default", {})
            extra_args = model_cfg.get("vllm_args", "")
            if extra_args:
                print(f"Using vllm_args from config: {extra_args}")
        except FileNotFoundError:
            extra_args = ""

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN env var not set")
        sys.exit(1)

    if not args.skip_install:
        print(f"Installing vLLM on {args.ip}...")
        install_vllm(ssh, venv_path=args.venv_path)

    print(f"Starting vLLM (model={args.model})...")
    start_vllm(
        ssh, model_id=args.model, hf_token=hf_token,
        port=args.port, extra_args=extra_args,
        venv_path=args.venv_path,
    )

    print(f"Waiting for vLLM to be ready (timeout={args.timeout}s)...")
    if not wait_for_vllm_ready(ssh, port=args.port, timeout=args.timeout):
        print("ERROR: vLLM did not become ready in time")
        sys.exit(1)

    print("vLLM is ready!")

    if args.tunnel:
        print(f"Opening SSH tunnel localhost:{args.local_port} -> {args.ip}:{args.port}")
        tunnel = ssh.open_tunnel(args.local_port, args.port)
        print(f"Tunnel open. vLLM available at http://localhost:{args.local_port}/v1")
        print("Press Ctrl+C to stop tunnel and exit.")

        def _cleanup(signum, frame):
            tunnel.terminate()
            tunnel.wait()
            print("\nTunnel closed.")
            sys.exit(0)

        signal.signal(signal.SIGINT, _cleanup)
        signal.signal(signal.SIGTERM, _cleanup)
        tunnel.wait()
    else:
        print(f"vLLM running on {args.ip}:{args.port}")
        print(f"To tunnel: ssh -i {args.key_file} -N -L {args.port}:localhost:{args.port} ubuntu@{args.ip}")


if __name__ == "__main__":
    main()
