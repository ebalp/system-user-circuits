#!/usr/bin/env python3
"""Start vLLM on a Lambda Cloud instance via SSH.

Installs vLLM in a virtualenv, starts the server, waits for readiness,
and opens an SSH tunnel so you can access it locally.

Usage:
    uv run python -m lambda_cloud.scripts.launch_vllm --ip 1.2.3.4 --model meta-llama/Llama-3.1-8B-Instruct
    uv run python -m lambda_cloud.scripts.launch_vllm --ip 1.2.3.4 --model meta-llama/Llama-3.1-8B-Instruct --tunnel
"""

import argparse
import logging
import os
import signal
import sys

from lambda_cloud.ssh import SSHConnection
from lambda_cloud.vllm_server import install_vllm, start_vllm, wait_for_vllm_ready, stop_vllm


def main():
    parser = argparse.ArgumentParser(description="Launch vLLM on a Lambda Cloud instance")
    parser.add_argument("--ip", required=True, help="Instance IP address")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--key-file", default="~/.ssh/anusha-cre-lambda-key.pem", help="SSH key file")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--venv-path", default="/home/ubuntu/vllm-venv", help="Virtualenv path on remote")
    parser.add_argument("--extra-args", default="", help="Extra vLLM serve arguments")
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

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN env var not set")
        sys.exit(1)

    ssh = SSHConnection(ip=args.ip, key_file=args.key_file)

    if not args.skip_install:
        print(f"Installing vLLM on {args.ip}...")
        install_vllm(ssh, venv_path=args.venv_path)

    print(f"Starting vLLM (model={args.model})...")
    start_vllm(
        ssh, model_id=args.model, hf_token=hf_token,
        port=args.port, extra_args=args.extra_args,
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
