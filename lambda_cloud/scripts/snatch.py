#!/usr/bin/env python3
"""Poll Lambda Cloud for GPU availability, launch, and optionally bootstrap.

Usage:
    # Snatch + bootstrap (auto-discovers .sync.env in repo root)
    uv run python -m lambda_cloud.scripts.snatch --setup

    # Explicit env file
    uv run python -m lambda_cloud.scripts.snatch --env-file myname.sync.env --setup

    # Snatch only (credentials already sourced)
    uv run python -m lambda_cloud.scripts.snatch

Checks every 10 seconds. Launches the first matching instance that has
capacity in any region. With --setup, also bootstraps the instance
(uploads .sync.env, configures GitHub credentials, clones repo, installs env).
"""

import argparse
import json
import logging
import os
import re
import sys
import time

import httpx
import yaml

import glob as _glob

BASE_URL = "https://cloud.lambda.ai/api/v1"
POLL_INTERVAL = 10  # seconds
IMAGE_FAMILY = "lambda-stack-22-04"

logger = logging.getLogger(__name__)


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


def _load_config(config_path: str) -> dict:
    """Load config YAML and return parsed dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_env_file(env_file: str) -> None:
    """Source shell export lines from an env file into os.environ.

    Only sets variables that are not already set in the environment,
    so explicit env vars or prior `source` take precedence.
    """
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r'^export\s+([A-Za-z_][A-Za-z_0-9]*)=(.*)', line)
            if m:
                key, value = m.group(1), m.group(2)
                # Strip surrounding quotes
                value = value.strip().strip('"').strip("'")
                if key not in os.environ and value:
                    os.environ[key] = value


def _get_api_key() -> str:
    key = os.environ.get("LAMBDA_API_KEY")
    if not key:
        print("ERROR: LAMBDA_API_KEY not set.")
        print("  Either: source <your>.sync.env")
        print("  Or:     --env-file <your>.sync.env")
        sys.exit(1)
    return key


def _auth(api_key: str) -> httpx.BasicAuth:
    return httpx.BasicAuth(username=api_key, password="")


def _find_image_for_region(api_key: str, region: str) -> str | None:
    """Find the Lambda Stack 22.04 image ID for a specific region."""
    try:
        resp = httpx.get(
            f"{BASE_URL}/images",
            auth=_auth(api_key),
            timeout=15,
        )
        resp.raise_for_status()
        images = resp.json().get("data", [])
        for img in images:
            if img.get("family") == IMAGE_FAMILY and img.get("region", {}).get("name") == region:
                return img.get("id")
    except Exception as e:
        print(f"  Warning: could not list images ({e}), will use default OS")
    return None


def _find_available_instance(api_key: str, target_instances: list[str]) -> tuple[str, str] | None:
    """Check for GPU availability. Returns (instance_type, region) or None."""
    resp = httpx.get(
        f"{BASE_URL}/instance-types",
        auth=_auth(api_key),
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json().get("data", {})

    # Show what's available
    available = [k for k, v in data.items() if v.get("regions_with_capacity_available")]
    if available:
        print(f"  Available types: {', '.join(available)}")

    for target in target_instances:
        info = data.get(target)
        if not info:
            continue
        regions = info.get("regions_with_capacity_available", [])
        if regions:
            region = regions[0].get("name")
            desc = info.get("instance_type", {}).get("description", target)
            price = info.get("instance_type", {}).get("price_cents_per_hour", "?")
            print(f"  FOUND: {target} ({desc}) in {region} @ ${price/100:.2f}/hr")
            return target, region

    return None


def _launch_instance(
    api_key: str,
    instance_type: str,
    region: str,
    ssh_key_name: str,
    image_id: str | None,
) -> dict:
    """Launch an instance and return the API response."""
    body: dict = {
        "region_name": region,
        "instance_type_name": instance_type,
        "ssh_key_names": [ssh_key_name],
        "file_system_names": [],
        "name": f"snatched-{instance_type}",
    }
    if image_id:
        body["image"] = {"id": image_id}

    resp = httpx.post(
        f"{BASE_URL}/instance-operations/launch",
        auth=_auth(api_key),
        json=body,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _poll_instance_ip(api_key: str, instance_id: str, timeout: int = 300) -> str:
    """Poll until instance has an IP address."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = httpx.get(
                f"{BASE_URL}/instances/{instance_id}",
                auth=_auth(api_key),
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})
            ip = data.get("ip")
            status = data.get("status", "unknown")
            if ip:
                return ip
            print(f"  Instance {instance_id}: status={status}, waiting for IP...")
        except Exception as e:
            print(f"  Error polling: {e}")
        time.sleep(10)
    raise TimeoutError(f"Instance {instance_id} didn't get an IP within {timeout}s")


def main():
    parser = argparse.ArgumentParser(description="Snatch a Lambda Cloud GPU instance")
    parser.add_argument(
        "--config", default="lambda_cloud/config/lambda.yaml",
        help="Path to lambda.yaml config",
    )
    parser.add_argument(
        "--env-file", type=str,
        help="Path to .sync.env file (default: auto-discover *.sync.env in repo root)",
    )
    parser.add_argument(
        "--setup", action="store_true",
        help="After launch, bootstrap the instance",
    )
    parser.add_argument(
        "--branch", default="main",
        help="Git branch to check out (with --setup)",
    )
    args = parser.parse_args()

    # Auto-discover env file if not specified
    if not args.env_file:
        args.env_file = _find_env_file()

    if args.setup and not args.env_file:
        parser.error("--setup requires an env file (--env-file or a single *.sync.env in repo root)")

    # Load env vars from file if provided (won't override already-set vars)
    if args.env_file:
        _load_env_file(args.env_file)

    config = _load_config(args.config)
    api_key = _get_api_key()

    ssh_key_name = config.get("ssh_key_name", "anusha-cre-lambda-key")
    ssh_key_file = config.get("ssh_key_file", "~/.ssh/anusha-cre-lambda-key.pem")
    target_instances = config.get("instance_preferences", ["gpu_1x_a100", "gpu_1x_a100_sxm4"])

    print("Snatching Lambda instance...")
    print(f"  Targets:       {', '.join(target_instances)}")
    print(f"  Image:         {IMAGE_FAMILY}")
    print(f"  Region:        any")
    print(f"  SSH key:       {ssh_key_name}")
    print(f"  Poll interval: {POLL_INTERVAL}s")
    if args.setup:
        print(f"  Setup:         yes (branch={args.branch})")
    print()

    attempt = 0
    while True:
        attempt += 1
        now = time.strftime("%H:%M:%S")
        try:
            result = _find_available_instance(api_key, target_instances)
            if result is None:
                print(f"[{now}] No matching GPU available (attempt {attempt})", end="\r")
                time.sleep(POLL_INTERVAL)
                continue

            instance_type, region = result

            print(f"  Looking up {IMAGE_FAMILY} image for {region}...")
            image_id = _find_image_for_region(api_key, region)
            if image_id:
                print(f"  Image ID: {image_id}")
            else:
                print(f"  {IMAGE_FAMILY} not found for {region}, using default OS")

            print(f"\n[{now}] Launching {instance_type} in {region}...")
            resp = _launch_instance(api_key, instance_type, region, ssh_key_name, image_id)
            instance_ids = resp.get("data", {}).get("instance_ids", [])
            if not instance_ids:
                print(f"  ERROR: no instance IDs in response: {json.dumps(resp)}")
                sys.exit(1)

            instance_id = instance_ids[0]
            print(f"  Instance ID: {instance_id}")

            print("  Waiting for IP address...")
            ip = _poll_instance_ip(api_key, instance_id)

            print("\n=== INSTANCE LAUNCHED ===")
            print(f"  ID:     {instance_id}")
            print(f"  IP:     {ip}")
            print(f"  Type:   {instance_type}")
            print(f"  Region: {region}")
            print(f"  Image:  {IMAGE_FAMILY}")
            print(f"  SSH:    ssh -i {ssh_key_file} ubuntu@{ip}")
            print("=========================")

            # ── Optional: bootstrap the instance ──
            if args.setup:
                from lambda_cloud.ssh import SSHConnection
                from lambda_cloud.instance_setup import bootstrap_instance

                logging.basicConfig(
                    level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S",
                )

                ssh = SSHConnection(ip=ip, key_file=ssh_key_file)
                print(f"\nWaiting for SSH on {ip}...")
                if not ssh.wait_for_ssh(timeout=300):
                    print(f"ERROR: SSH not reachable on {ip} after 300s")
                    sys.exit(1)

                print("Bootstrapping instance...")
                bootstrap_instance(ssh, env_file_path=args.env_file, branch=args.branch)
                print(f"\n=== INSTANCE READY ===")
                print(f"  SSH:  ssh -i {ssh_key_file} ubuntu@{ip}")
                print(f"  Repo: /home/ubuntu/system-user-circuits")
                print("======================")

            break

        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            print(f"\n[{now}] API error: {e.response.status_code} - {error_body}")
            if e.response.status_code == 401:
                print("Check your LAMBDA_API_KEY")
                sys.exit(1)
            if "insufficient-capacity" in error_body:
                print("  Capacity gone, continuing to poll...")
                time.sleep(POLL_INTERVAL)
                continue
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nStopped.")
            sys.exit(0)


if __name__ == "__main__":
    main()
