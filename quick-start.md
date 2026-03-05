# Quick Start

## Prerequisites

1. **SSH key** — Either get the shared Lambda SSH key (`anusha-cre-lambda-key.pem`) and place it at `~/.ssh/anusha-cre-lambda-key.pem`, or create your own key in the [Lambda Cloud console](https://cloud.lambda.ai) under **SSH Keys** and update `ssh_key_name` and `ssh_key_file` in `lambda_cloud/config/lambda.yaml`. All scripts read the key from this config.

2. **Personal bucket** — In the [Lambda Cloud console](https://cloud.lambda.ai):
   - Create a bucket under **Filesystem → S3 Adapter Filesystems** (one per person, Washington or Ohio region). Works with instances from any region.
   - Generate S3 credentials under **Filesystem → S3 Adapter Keys**.
   - Generate a GitHub token at **GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)** with `repo` scope.

3. **Credentials file** — Copy `sync.env.template` to `.sync.env` and fill in your values. If you want Claude Code to use OpenRouter, uncomment the OpenRouter block. This file is gitignored.

4. **SSH config** — Add an entry to `~/.ssh/config`:
   ```
   Host lambda
       HostName <instance-ip>
       User ubuntu
       IdentityFile ~/.ssh/anusha-cre-lambda-key.pem
   ```
   Update `HostName` each time you get a new instance. If you have Claude Code locally, you can run `/update-ssh-lambda <new-ip>` to update it automatically.

## Launch a new instance

```bash
uv run python -m lambda_cloud.scripts.snatch --setup
```

This will:
- Auto-discover your `.sync.env` file in the repo root
- Poll Lambda Cloud every 10s until a GPU is available
- Launch the instance
- Bootstrap it: upload credentials, clone repo, install Python env, install Claude Code
- `.sync.env` is auto-sourced on login (OpenRouter, HF_TOKEN, etc.)

## Bootstrap an existing instance

If you already have a running instance (e.g. launched from the Lambda dashboard):

```bash
uv run python -m lambda_cloud.scripts.setup_instance --ip <instance-ip>
```

Then update your SSH config with the new IP (`/update-ssh-lambda <ip>` if you have Claude Code locally).

## Connect to the instance

- **SSH** — `ssh lambda`
- **VS Code** — Install [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh), then `F1` → `Remote-SSH: Connect to Host…` → `lambda`
- **JupyterLab** — Open the Lambda Cloud dashboard and click the Jupyter button for your instance

## After bootstrap

1. **Authenticate Claude Code** — `.sync.env` is auto-sourced on login, so if you configured OpenRouter it works automatically. Just SSH in and run `claude`.

2. **Download data from bucket** (if you have previous work):
   ```bash
   ./lambda-sync.sh download
   ```

## Data sync

**Code goes in GitHub. Data goes in the bucket.** The local instance disk (512GB) is ephemeral — always upload results before terminating.

```bash
./lambda-sync.sh upload              # push results to bucket
./lambda-sync.sh download            # restore data from bucket
./lambda-sync.sh upload phase0/data  # sync specific paths
```

By default, `upload` syncs all `*/data/` and `*/reports/` directories. `download` restores the full bucket. Both ask for confirmation. Patterns in `.syncignore` are excluded.
