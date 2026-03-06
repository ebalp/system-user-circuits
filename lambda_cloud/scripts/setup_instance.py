#!/usr/bin/env python3
"""Bootstrap a Lambda Cloud instance via SSH.

Usage:
    uv run python -m lambda_cloud.scripts.setup_instance --ip 1.2.3.4
    uv run python -m lambda_cloud.scripts.setup_instance --ip 1.2.3.4 --env-file myname.sync.env
    uv run python -m lambda_cloud.scripts.setup_instance --ip 1.2.3.4 --branch dev
"""

import argparse
import glob as _glob
import logging
import os

import yaml

from lambda_cloud.instance_setup import bootstrap_instance, _remote_dir_from_url
from lambda_cloud.ssh import SSHConnection

DEFAULT_CONFIG = "lambda_cloud/config/lambda.yaml"


def _find_env_file() -> str | None:
    """Auto-discover a .sync.env file in the repo root."""
    if os.path.isfile(".sync.env"):
        return ".sync.env"
    matches = _glob.glob("*.sync.env")
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"Multiple .sync.env files found: {', '.join(sorted(matches))}")
        print("Use --env-file to pick one.")
        return None
    return None


def main():
    parser = argparse.ArgumentParser(description="Bootstrap a Lambda Cloud instance")
    parser.add_argument("--ip", required=True, help="Instance IP address")
    parser.add_argument(
        "--env-file",
        help="Local .sync.env file to upload (default: auto-discover *.sync.env in repo root)",
    )
    parser.add_argument("--key-file", default=None, help="SSH key file (default: from lambda.yaml)")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to lambda.yaml config")
    parser.add_argument("--branch", default="main", help="Git branch to check out")
    parser.add_argument("--wait-ssh", type=int, default=300, help="Seconds to wait for SSH")
    args = parser.parse_args()

    # Load config
    try:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        config = {}

    if not args.key_file:
        args.key_file = config.get("ssh_key_file", "~/.ssh/id_rsa")

    if not args.env_file:
        args.env_file = _find_env_file()
    if not args.env_file:
        parser.error("No env file found. Use --env-file or place a single *.sync.env in the repo root.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    ssh = SSHConnection(ip=args.ip, key_file=args.key_file)

    print(f"Waiting for SSH on {args.ip}...")
    if not ssh.wait_for_ssh(timeout=args.wait_ssh):
        print(f"ERROR: SSH not reachable on {args.ip} after {args.wait_ssh}s")
        raise SystemExit(1)

    repo_url = config.get("repo_url")
    if not repo_url:
        print("ERROR: repo_url must be set in config")
        raise SystemExit(1)
    remote_dir = config.get("repo_dir") or _remote_dir_from_url(repo_url)

    bootstrap_instance(
        ssh, env_file_path=args.env_file, repo_url=repo_url,
        branch=args.branch, remote_dir=remote_dir,
    )
    print(f"Instance {args.ip} bootstrapped successfully.")


if __name__ == "__main__":
    main()
