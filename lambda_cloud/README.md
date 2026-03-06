# lambda_cloud

Automates the lifecycle of Lambda Cloud GPU instances and vLLM inference servers. Handles launching instances, bootstrapping them (repo, credentials, Python env, Claude Code), deploying vLLM, and tearing everything down safely.

This module can be used independently of the parent project — it has no project-specific imports.

> **Implementation notes:** vLLM is installed in an isolated virtualenv (avoids Lambda Stack's system TensorFlow/numpy conflicts). Health checks and inference use SSH tunnels (Lambda's firewall blocks port 8000 externally). All setup is done over SSH rather than cloud-init for reliability and visibility.

## Getting Started

1. Copy the example config and fill in your values:
   ```bash
   cp lambda_cloud/config/lambda.yaml.example lambda_cloud/config/lambda.yaml
   ```

2. Set required environment variables (or put them in `.sync.env`):
   ```bash
   export LAMBDA_API_KEY="..."   # Lambda Cloud Dashboard -> Settings -> API Keys
   export HF_TOKEN="..."         # Hugging Face token for gated models
   ```

3. Configure your SSH key:
   - Register it in Lambda Cloud Dashboard -> SSH Keys
   - Set `ssh_key_name` and `ssh_key_file` in `lambda.yaml`

4. Set `repo_url` in `lambda.yaml` to your git repo (used by `--setup` bootstrap)

## Modules

| Module | Purpose |
|--------|---------|
| `config.py` | `LambdaConfig`, `LambdaInstance` dataclasses, `load_lambda_config()` YAML loader |
| `manager.py` | `LambdaCloudManager` — launch, terminate, poll for IP, safety nets (atexit + signals) |
| `ssh.py` | `SSHConnection` — run commands, background processes, tunnels, SCP uploads (no paramiko) |
| `vllm_server.py` | Install vLLM in venv, start/stop server, SSH-proxied health checks, tunnel readiness, `ensure_vllm_running()` orchestrator |
| `instance_setup.py` | Bootstrap: upload credentials, clone repo, install env, optionally install Claude Code |

## CLI Scripts

```bash
# Snatch a GPU and bootstrap (auto-discovers .sync.env)
uv run python -m lambda_cloud.scripts.snatch --setup

# Bootstrap an existing instance
uv run python -m lambda_cloud.scripts.setup_instance --ip <ip>

# Check vLLM status
uv run python -m lambda_cloud.scripts.launch_vllm --ip <ip> --status

# Stop vLLM
uv run python -m lambda_cloud.scripts.launch_vllm --ip <ip> --stop

# Launch vLLM (reads vllm_args from config)
uv run python -m lambda_cloud.scripts.launch_vllm --ip <ip> --model meta-llama/Llama-3.1-8B-Instruct --tunnel
```

## Python API

```python
from lambda_cloud.config import load_lambda_config
from lambda_cloud.manager import LambdaCloudManager
from lambda_cloud.ssh import SSHConnection
from lambda_cloud.vllm_server import ensure_vllm_running, wait_for_vllm_through_tunnel

# Launch and manage an instance
config = load_lambda_config("lambda_cloud/config/lambda.yaml", model_id)
with LambdaCloudManager(config) as mgr:
    ssh = SSHConnection(ip=mgr.instance.ip)
    ssh.wait_for_ssh()
    ensure_vllm_running(
        ssh, model_id=config.model_id, hf_token=config.hf_token,
        port=config.vllm_port, extra_args=config.vllm_extra_args,
        venv_path=config.vllm_venv_path,
        readiness_timeout=config.readiness_timeout,
    )
    tunnel, local_port = ssh.open_tunnel(config.vllm_port, config.vllm_port)
    vllm_url = f"http://localhost:{local_port}/v1"
    wait_for_vllm_through_tunnel(vllm_url, config.model_id)
    # ... run inference against vllm_url ...
    tunnel.terminate()
# instance auto-terminated on exit
```

### vllm_server

```python
from lambda_cloud.vllm_server import (
    ensure_vllm_running,          # check status -> install -> start -> wait (one call)
    wait_for_vllm_through_tunnel, # poll /v1/models through local SSH tunnel, validate model
    install_vllm, start_vllm,     # low-level: install in venv, start in background
    wait_for_vllm_ready,          # low-level: SSH-proxied curl health check
    vllm_status, stop_vllm,       # check/stop running vLLM process
)

# High-level: ensures vLLM is running (skips if already up)
ensure_vllm_running(ssh, model_id="meta-llama/Llama-3.1-8B-Instruct", hf_token="...")

# After opening a tunnel, validate the model is reachable locally
wait_for_vllm_through_tunnel("http://localhost:8000/v1", "meta-llama/Llama-3.1-8B-Instruct")
```

### SSHConnection

```python
ssh = SSHConnection(ip="1.2.3.4")
ssh.wait_for_ssh(timeout=300)           # wait for instance to be reachable
ssh.run("nvidia-smi")                   # run command, raises on failure
ssh.run_background("python train.py")   # nohup + disown
ssh.upload_file("local.txt", "/home/ubuntu/remote.txt")
tunnel = ssh.open_tunnel(8000, 8000)    # local port forwarding
```

### Two modes: managed vs existing instance

**Managed (auto-launch + auto-terminate):** Use `LambdaCloudManager` as a context manager. It launches an instance, runs your code, and terminates the instance when done. Safety nets ensure the instance is always terminated — even on Ctrl+C, crashes, or unhandled exceptions — via `__exit__`, `atexit`, and SIGINT/SIGTERM handlers.

**Existing instance:** Use `--ip` to connect to an instance you've already launched (e.g. via `snatch`). The instance is never terminated automatically — you manage its lifecycle yourself. This is the mode used by `launch_vllm.py` and `setup_instance.py`.

## Common Workflows

All commands run **locally** (not on the Lambda instance). They SSH into the instance under the hood. Env vars must be sourced locally first:

```bash
source .sync.env  # sets LAMBDA_API_KEY, HF_TOKEN
```

### Launch vLLM manually (for interactive use)

```bash
uv run python -m lambda_cloud.scripts.launch_vllm --ip <ip> \
  --model meta-llama/Llama-3.1-8B-Instruct --tunnel
```

This opens an SSH tunnel so you can query `http://localhost:8000/v1` locally.

## Configuration

`lambda_cloud/config/lambda.yaml` — copy from `lambda.yaml.example` and customize:

```yaml
ssh_key_name: "my-lambda-key"
ssh_key_file: "~/.ssh/my-lambda-key.pem"
repo_url: "https://github.com/your-org/your-repo.git"
# repo_dir: "/home/ubuntu/your-repo"  # derived from repo_url if omitted

instance_preferences: [gpu_1x_a100, gpu_1x_a100_sxm4]

defaults:
  region: us-east-1
  vllm_port: 8000
  vllm_venv_path: /home/ubuntu/vllm-venv
  concurrent_per_model: 10  # inference concurrency

model_gpu_map:
  meta-llama/Llama-3.1-8B-Instruct:
    instance_type: gpu_1x_a10
    vllm_args: "--max-model-len 4096"
  meta-llama/Llama-3.3-70B-Instruct:
    instance_type: gpu_8x_a100
    vllm_args: "--tensor-parallel-size 8"
```

| Field | Required | Description |
|-------|----------|-------------|
| `ssh_key_name` | Yes | SSH key name registered in Lambda Cloud |
| `ssh_key_file` | Yes | Local path to the private key file |
| `repo_url` | For `--setup` | Git repo URL to clone on the instance |
| `repo_dir` | No | Remote clone directory (derived from `repo_url` if omitted) |
| `instance_preferences` | No | GPU instance types to try, in priority order |
| `model_gpu_map` | No | Per-model GPU type and vLLM args |

## Tests

```bash
# Unit tests
uv run pytest lambda_cloud/tests/ -v -m "not live"

# Live tests (requires vLLM running on ssh lambda)
# Auto-detects the served model; override with VLLM_MODEL_ID env var
uv run pytest lambda_cloud/tests/test_live.py -v -m live
```
