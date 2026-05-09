---
name: lambda-gpu
description: "Reference guide for the lambda-cloud-toolkit package — both the lambda-gpu CLI and the Python API. Use this skill whenever the user asks about Lambda Cloud GPU instances, launching or snatching GPUs, deploying vLLM on remote instances, syncing data to/from Lambda Cloud Filesystem, bootstrapping Lambda instances, SSH connections to Lambda, updating the SSH config for a new Lambda instance IP, or writing Python code that uses lambda_cloud_toolkit (LambdaCloudManager, SSHConnection, LambdaStorage, vllm_server functions). Also trigger when users mention lambda-gpu commands (snatch, setup, vllm, sync), ask how to run experiments on Lambda Cloud, or need to configure SSH for a newly launched instance."
---

# lambda-gpu CLI Reference

The `lambda-gpu` command (from the `lambda-cloud-toolkit` package) manages the full lifecycle of Lambda Cloud GPU instances: launching, bootstrapping, deploying vLLM, and syncing data.

## Typical Workflow

The commands are designed to be used in sequence:

```
snatch (get a GPU) → setup (bootstrap it) → vllm (deploy model) → [run experiments] → sync upload (save results)
```

`snatch --setup` combines the first two steps. After experiments, always sync upload before terminating.

---

## Commands

### 1. `lambda-gpu snatch` — Poll for GPU availability and launch

Repeatedly polls Lambda Cloud until a GPU becomes available, then launches it.

```bash
# Poll with config defaults, launch when available
lambda-gpu snatch

# Poll, launch, AND bootstrap the instance automatically
lambda-gpu snatch --setup

# Target specific GPU type(s)
lambda-gpu snatch --gpu gpu_1x_a100,gpu_1x_a100_sxm4

# Setup with a specific branch
lambda-gpu snatch --setup --branch feature-branch
```

| Flag | Description |
|------|-------------|
| `--gpu TYPE[,TYPE]` | GPU type(s), comma-separated. Overrides `instance_preferences` from config |
| `--setup` | Bootstrap instance after launch (clone repo, install deps, etc.) |
| `--branch BRANCH` | Git branch to check out during setup (default: `main`) |
| `--config PATH` | Path to `lambda-cloud.yaml` |
| `--env-file PATH` | Path to `.sync.env` credentials file |

The command prints the instance ID, IP address, and SSH command when it succeeds.

---

### 2. `lambda-gpu setup --ip IP` — Bootstrap an existing instance

Run this on an instance that's already running but hasn't been set up yet (e.g., you launched it from the Lambda Cloud console).

```bash
lambda-gpu setup --ip 192.0.2.10

# With specific SSH key and branch
lambda-gpu setup --ip 192.0.2.10 --key-file ~/.ssh/lambda.pem --branch dev
```

| Flag | Description |
|------|-------------|
| `--ip IP` | Instance IP address (required) |
| `--branch BRANCH` | Git branch to check out (default: `main`) |
| `--key-file PATH` | SSH private key file |
| `--wait-ssh SECS` | Timeout waiting for SSH to become reachable (default: 300) |
| `--config PATH` | Path to `lambda-cloud.yaml` |
| `--env-file PATH` | Path to `.sync.env` |

**What bootstrap does** (in order):
1. Uploads `.sync.env` and configures GitHub token for private repo access
2. Clones the project repo (and any `dependency_repos` from config)
3. Configures git identity from env vars
4. Installs uv and Python 3.12
5. Runs `uv sync` to set up the project environment
6. Runs optional `setup_script` from config
7. Auto-sources `.sync.env` on SSH login
8. Installs Claude Code

---

### 3. `lambda-gpu vllm --ip IP` — Deploy and manage vLLM

Launch, check, or stop a vLLM inference server on a remote instance.

```bash
# Deploy a model and open an SSH tunnel to access it locally
lambda-gpu vllm --ip 192.0.2.10 --model meta-llama/Llama-3.1-8B-Instruct --tunnel

# Check if vLLM is running
lambda-gpu vllm --ip 192.0.2.10 --status

# Stop the server
lambda-gpu vllm --ip 192.0.2.10 --stop

# Custom vLLM args (overrides config model_gpu_map)
lambda-gpu vllm --ip 192.0.2.10 --model meta-llama/Llama-3.1-8B-Instruct \
  --extra-args "--max-model-len 4096 --enforce-eager"

# Skip install if vLLM is already installed
lambda-gpu vllm --ip 192.0.2.10 --model meta-llama/Llama-3.1-8B-Instruct --skip-install --tunnel
```

| Flag | Description |
|------|-------------|
| `--ip IP` | Instance IP address (required) |
| `--model MODEL` | HuggingFace model ID (required for launch) |
| `--status` | Check if vLLM is running (mutually exclusive with launch/stop) |
| `--stop` | Stop the vLLM server |
| `--tunnel` | Open SSH tunnel after launch (access vLLM at localhost) |
| `--local-port PORT` | Local port for the tunnel (default: 8000) |
| `--port PORT` | Remote vLLM port (default: 8000) |
| `--extra-args ARGS` | Extra arguments passed to `vllm serve` |
| `--skip-install` | Skip `pip install vllm` (use if already installed) |
| `--timeout SECS` | How long to wait for vLLM to become ready (default: 900) |
| `--venv-path PATH` | Remote virtualenv path (default: `/home/ubuntu/vllm-venv`) |

**How it works:** vLLM is installed in a separate virtualenv on the instance. The server runs in the background via `nohup` so it survives SSH disconnects. Readiness is checked via SSH-proxied curl (Lambda's firewall blocks port 8000 externally). With `--tunnel`, an SSH port forward is opened so you can hit `http://localhost:8000/v1` locally.

---

### 4. `lambda-gpu sync` — S3-compatible data sync

Upload, download, or list files on Lambda Cloud Filesystem (S3-compatible storage).

```bash
# Upload default sync_dir from config
lambda-gpu sync upload

# Upload specific paths
lambda-gpu sync upload phase0_v2/data/results phase0_v2/data/logs

# Download default sync_dir
lambda-gpu sync download

# Download specific path
lambda-gpu sync download phase0_v2/data

# List bucket contents
lambda-gpu sync ls
lambda-gpu sync ls data/results/
```

| Subcommand | Description |
|------------|-------------|
| `upload [PATHS...]` | Upload local directories to bucket. Defaults to `sync_dir` from config |
| `download [PATHS...]` | Download from bucket to local. Defaults to `sync_dir` from config |
| `ls [PATH]` | List bucket contents at the given path (or root) |

Shared flags: `--config PATH`, `--env-file PATH`

**Important:** Upload overwrites bucket contents; download overwrites local files. Uses `aws s3 sync` under the hood. Exclusion patterns come from `.syncignore` (works like `.gitignore`).

---

## Configuration

### lambda-cloud.yaml

Auto-discovered in the current directory (or specify with `--config`).

```yaml
ssh_key_name: "my-key"
ssh_key_file: "~/.ssh/my-key.pem"
repo_url: "https://github.com/org/repo.git"

defaults:
  vllm_port: 8000
  poll_interval: 10          # seconds between snatch polls
  max_launch_retries: 5      # retries on capacity race
  readiness_timeout: 900     # seconds for vLLM to load

# GPU preferences (tried in order during snatch)
instance_preferences:
  - gpu_1x_a100
  - gpu_1x_a100_sxm4

# Per-model GPU and vLLM settings
model_gpu_map:
  meta-llama/Llama-3.1-8B-Instruct:
    instance_type: gpu_1x_a100
    instance_preferences: [gpu_1x_a100, gpu_1x_a100_sxm4]
    vllm_args: "--max-model-len 4096"
  meta-llama/Llama-3.3-70B-Instruct:
    instance_type: gpu_8x_a100
    vllm_args: "--tensor-parallel-size 8 --max-model-len 4096"
  _default:
    instance_type: gpu_1x_a100
    vllm_args: "--max-model-len 4096"

# S3-compatible storage
storage:
  sync_dir: "data"
  syncignore: ".syncignore"
```

### .sync.env (credentials)

Auto-discovered in the current directory (or specify with `--env-file`). Required variables:

| Variable | Used by |
|----------|---------|
| `LAMBDA_API_KEY` | Instance management (snatch, etc.) |
| `HF_TOKEN` | Gated model downloads (vLLM) |
| `BUCKET_NAME` | Data sync |
| `LAMBDA_ACCESS_KEY_ID` | Data sync |
| `LAMBDA_SECRET_ACCESS_KEY` | Data sync |
| `GITHUB_TOKEN` | Private repo cloning (setup) |
| `GIT_USER_NAME` | Git config on instance |
| `GIT_USER_EMAIL` | Git config on instance |

---

## GPU Sizing Guide

| Model size | Minimum VRAM | Recommended instance | Cost |
|-----------|-------------|---------------------|------|
| 7-8B | ~16 GB | `gpu_1x_a100` (40 GB) | ~$1.48/hr |
| 70B | ~140 GB | `gpu_8x_a100` (8x40 GB) | ~$11.84/hr |

Lambda A100s are 40 GB (not 80 GB). The only 80 GB A100 is `gpu_8x_a100_80gb_sxm4`. 70B models need tensor parallelism (`--tensor-parallel-size 8`).

Rule of thumb for BF16 inference: ~2 GB VRAM per 1B parameters.

---

## Lambda Cloud API

**Base URL**: `https://cloud.lambdalabs.com/api/v1`
**Auth**: Bearer token or Basic auth with `LAMBDA_API_KEY`

```bash
source .sync.env
curl -u "$LAMBDA_API_KEY:" https://cloud.lambdalabs.com/api/v1/instances
```

### Querying available instance types

The `/instance-types` endpoint returns all GPU types with specs, pricing, and real-time availability per region:

```bash
# List all instance types with description and price
curl -u "$LAMBDA_API_KEY:" https://cloud.lambdalabs.com/api/v1/instance-types \
  | python3 -c "import sys,json; [print(f'{k:35} {v[\"instance_type\"][\"description\"]:45} \${v[\"instance_type\"][\"price_cents_per_hour\"]/100:.2f}/hr') for k,v in sorted(json.load(sys.stdin)['data'].items())]"

# Check which regions have capacity right now
curl -u "$LAMBDA_API_KEY:" https://cloud.lambdalabs.com/api/v1/instance-types \
  | python3 -c "import sys,json; [print(f'{k}: {v[\"regions_with_capacity_available\"]}') for k,v in sorted(json.load(sys.stdin)['data'].items()) if v['regions_with_capacity_available']]"
```

Each instance type includes: `name`, `description`, `gpu_description`, `price_cents_per_hour`, and `specs` (vcpus, memory_gib, storage_gib, gpus).

Instance type names follow the pattern: `gpu_{count}x_{gpu_model}[_{variant}]`

### Querying available images

The `/images` endpoint lists OS images available for launching instances:

```bash
# List available images
curl -u "$LAMBDA_API_KEY:" https://cloud.lambdalabs.com/api/v1/images \
  | python3 -c "import sys,json; [print(f'{img[\"name\"]:30} {img[\"family\"]:20} {img[\"architecture\"]:10} {img[\"region\"][\"name\"]}') for img in json.load(sys.stdin)['data']]"
```

Each image includes: `id`, `name`, `description`, `family` (e.g., `ubuntu-lts`), `version`, `architecture` (x86_64/arm64), and `region`. You can specify an image when launching via `ImageSpecificationID` (by id) or `ImageSpecificationFamily` (by family name).

### Other useful endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/instances` | List running instances |
| GET | `/instances/{id}` | Instance details (id, ip, status, type) |
| POST | `/instance-operations/launch` | Launch instance |
| POST | `/instance-operations/terminate` | Terminate instance |
| GET | `/ssh-keys` | List SSH keys |
| GET | `/file-systems` | List filesystems |
| GET | `/firewall-rules` | List firewall rules |

**Rate limits**: 1 req/s general, 1 req/12s for launch (5/min).

For the full OpenAPI spec, see `../lambda-cloud-toolkit/docs/lambda-cloud-openapi.json`.

---

## Python API

For programmatic use (e.g., in experiment scripts), the key classes live in `lambda_cloud_toolkit`. Source code is at `../lambda-cloud-toolkit/src/lambda_cloud_toolkit/`.

### LambdaCloudManager — Instance lifecycle

```python
from lambda_cloud_toolkit import LambdaCloudManager, load_lambda_config

config = load_lambda_config("lambda-cloud.yaml", model_id="meta-llama/Llama-3.1-8B-Instruct")

# Context manager auto-terminates on exit
with LambdaCloudManager(config) as manager:
    ip = manager.instance.ip
    base_url = manager.get_base_url()  # http://{ip}:{port}/v1
    # ... run experiments ...

# Or manual lifecycle
manager = LambdaCloudManager(config)
instance = manager.launch()
manager.terminate()

# Check GPU availability
available = manager.list_available()
```

Source: `manager.py`, config types in `config.py`

### SSHConnection — Remote commands

```python
from lambda_cloud_toolkit.ssh import SSHConnection

ssh = SSHConnection(ip="192.0.2.10", key_file="~/.ssh/lambda.pem")
ssh.wait_for_ssh(timeout=300)
result = ssh.run("nvidia-smi", timeout=30)
ssh.run_background("python train.py")
ssh.upload_file("local.csv", "/home/ubuntu/data.csv")
tunnel, port = ssh.open_tunnel(8000, 8000)
```

Source: `ssh.py`

### vLLM server functions

```python
from lambda_cloud_toolkit.vllm_server import (
    ensure_vllm_running, install_vllm, start_vllm,
    stop_vllm, vllm_status, wait_for_vllm_ready,
)

ensure_vllm_running(ssh, model_id="...", hf_token="...")  # all-in-one
status = vllm_status(ssh, port=8000)  # returns {pid, model, cmdline} or None
```

Source: `vllm_server.py`

### LambdaStorage — S3 sync

```python
from lambda_cloud_toolkit.storage import LambdaStorage

storage = LambdaStorage(bucket_name="...", access_key_id="...", secret_access_key="...")
storage.upload("data/results", subpath="data/results")
storage.download("data/results", subpath="data/results")
storage.ls("data/")
```

Source: `storage.py`

### bootstrap_instance — Setup automation

```python
from lambda_cloud_toolkit.instance_setup import bootstrap_instance

bootstrap_instance(ssh, env_file_path=".sync.env", repo_url="...", branch="main",
                   remote_dir="/home/ubuntu/repo", dependency_repos=["..."])
```

Source: `instance_setup.py`

### Where to find more

| What | Where |
|------|-------|
| Full source | `../lambda-cloud-toolkit/src/lambda_cloud_toolkit/` |
| CLI implementation | `cli.py` |
| Config dataclasses | `config.py` |
| Lambda Cloud REST API | `../lambda-cloud-toolkit/docs/lambda-cloud-api-reference.md` |
| Config example | `../lambda-cloud-toolkit/lambda-cloud.yaml` |
| Credentials template | `../lambda-cloud-toolkit/examples/env.template` |
| Tests | `../lambda-cloud-toolkit/tests/` |

---

## SSH Configuration

When a new instance is launched, update the local SSH `lambda` host alias so `ssh lambda` points to the new IP. Steps:

1. Read `~/.ssh/config` to find the current `HostName` under `Host lambda`
2. Remove the old IP from `~/.ssh/known_hosts`: `ssh-keygen -R <old-ip>`
3. Remove the new IP from `~/.ssh/known_hosts` if present: `ssh-keygen -R <new-ip>`
4. Update `~/.ssh/config` — change the `HostName` under `Host lambda` to the new IP
5. Test: `ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=accept-new lambda echo "connection ok"`

If the test fails with "Operation timed out", the instance is likely still booting — wait and retry.

---

## Troubleshooting

**"No capacity" during snatch**: Normal — Lambda GPUs are scarce. The command keeps polling. Try adding more GPU types with `--gpu` to broaden availability.

**vLLM timeout**: The model may still be downloading. Check with `--status` — if the process is alive, it's likely still loading. Increase `--timeout` for large models.

**SSH connection refused**: New instances take 1-2 minutes for SSH to come up. `setup` waits automatically (up to `--wait-ssh` seconds).

**Sync fails**: Check that `LAMBDA_ACCESS_KEY_ID`, `LAMBDA_SECRET_ACCESS_KEY`, and `BUCKET_NAME` are set in `.sync.env`. The `aws` CLI must be available (pre-installed on Lambda instances; install locally with `pip install awscli`).

**Port 8000 not accessible**: Lambda's firewall blocks external access to port 8000. Use `--tunnel` to create an SSH port forward, then access vLLM at `http://localhost:8000/v1`.
