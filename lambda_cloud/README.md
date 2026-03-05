# lambda_cloud

Automates the lifecycle of Lambda Cloud GPU instances and vLLM inference servers. Handles launching instances, bootstrapping them (repo, credentials, Python env, Claude Code), deploying vLLM, and tearing everything down safely. Used across all phases.

> **Implementation notes:** vLLM is installed in an isolated virtualenv (avoids Lambda Stack's system TensorFlow/numpy conflicts). Health checks and inference use SSH tunnels (Lambda's firewall blocks port 8000 externally). All setup is done over SSH rather than cloud-init for reliability and visibility.

## Modules

| Module | Purpose |
|--------|---------|
| `config.py` | `LambdaConfig`, `LambdaInstance` dataclasses, `load_lambda_config()` YAML loader |
| `manager.py` | `LambdaCloudManager` — launch, terminate, poll for IP, safety nets (atexit + signals) |
| `ssh.py` | `SSHConnection` — run commands, background processes, tunnels, SCP uploads (no paramiko) |
| `vllm_server.py` | Install vLLM in venv, start/stop server, SSH-proxied health checks |
| `instance_setup.py` | Bootstrap: upload credentials, clone repo, install env, install Claude Code |

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
from lambda_cloud import LambdaCloudManager, load_lambda_config
from lambda_cloud.ssh import SSHConnection
from lambda_cloud.vllm_server import install_vllm, start_vllm, wait_for_vllm_ready

# Launch and manage an instance
config = load_lambda_config("lambda_cloud/config/lambda.yaml", model_id)
with LambdaCloudManager(config) as mgr:
    ssh = SSHConnection(ip=mgr.instance.ip)
    install_vllm(ssh)
    start_vllm(ssh, model_id=config.model_id, hf_token=config.hf_token)
    wait_for_vllm_ready(ssh)
    tunnel = ssh.open_tunnel(local_port=8000, remote_port=8000)
    # ... run experiments against http://localhost:8000/v1 ...
    tunnel.terminate()
# instance auto-terminated on exit
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

### Safety nets

`LambdaCloudManager` ensures instances are always terminated — even on crashes:
- Context manager `__exit__`
- `atexit` handler
- SIGINT/SIGTERM signal handlers

## Configuration

`lambda_cloud/config/lambda.yaml` supports per-model GPU mappings:

```yaml
ssh_key_name: "anusha-cre-lambda-key"
ssh_key_file: "~/.ssh/anusha-cre-lambda-key.pem"
instance_preferences: [gpu_1x_a100, gpu_1x_a100_sxm4]

defaults:
  region: us-east-1
  vllm_port: 8000
  vllm_venv_path: /home/ubuntu/vllm-venv

model_gpu_map:
  meta-llama/Llama-3.1-8B-Instruct:
    instance_type: gpu_1x_a10
    vllm_args: "--max-model-len 4096"
  meta-llama/Llama-3.3-70B-Instruct:
    instance_type: gpu_8x_a100
    vllm_args: "--tensor-parallel-size 8"
```

## Tests

```bash
# Unit tests
uv run pytest lambda_cloud/tests/ -v -m "not live"

# Live tests (requires vLLM running on ssh lambda)
uv run pytest lambda_cloud/tests/test_live.py -v -m live
```
