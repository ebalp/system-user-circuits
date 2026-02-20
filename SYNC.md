# Lambda Data Sync

Persist experiment data across Lambda AI instances and regions.

**Code goes in GitHub. Data goes in the bucket.**

The bucket stores only outputs that shouldn't be in git but need to persist: experiment results, generated reports, and similar data files. Everything else lives in the repo.

The sync script uploads `*/data/` and `*/reports/` directories to the bucket, and downloads them back when you start a new instance. You can also sync specific paths. Code files are never touched. Patterns in `.syncignore` are excluded from every sync.

## One-time account setup

In the [Lambda Cloud console](https://cloud.lambda.ai):

1. **Create a personal bucket** under **Filesystem → S3 Adapter Filesystems**. One bucket per person, only Washington and Ohio regions support this. Works with instances from any region. No instance filesystem needed.
2. **Get S3 credentials** under **Filesystem → S3 Adapter Keys**. Generate an access key.
3. **Get a GitHub token** under **GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)** with `repo` scope.

Then create your config file:

```bash
cp sync.env.template config.sync.env   # or any name ending in .sync.env
```

Fill in your details. This file is gitignored — keep it in the repo directory and upload it manually to each new instance (e.g. via Jupyter).

Seed your bucket:

```bash
./lambda-sync.sh config.sync.env setup
./lambda-sync.sh config.sync.env upload
```

## Daily workflow

### New instance

No filesystem attachment needed — instances have 512GB local disk.

```bash
cd /home/ubuntu
git clone https://github.com/ebalp/system-user-circuits.git
cd system-user-circuits
# upload your .sync.env to this directory via Jupyter
./lambda-sync.sh config.sync.env setup     # git config + uv env (all automatic)
./lambda-sync.sh config.sync.env download  # restore data from bucket
```

### Before shutting down

```bash
./lambda-sync.sh config.sync.env upload
```

### Using your HF token

Source your config to load `HF_API_KEY` into your shell:

```bash
source config.sync.env
```

## Script reference

```
./lambda-sync.sh <config-file> <mode> [path ...]
```

| Mode       | What it does                                                              |
|------------|---------------------------------------------------------------------------|
| `setup`    | Configures git, installs uv, sets up Python environment                   |
| `upload`   | Syncs `*/data/` and `*/reports/` to bucket (or explicit paths if given)   |
| `download` | Syncs bucket to local repo (or explicit paths if given)                   |

Both `upload` and `download` ask for one confirmation. Read the direction carefully — upload overwrites the bucket, download overwrites local data.

Pass one or more paths to sync only specific directories:

```bash
./lambda-sync.sh config.sync.env upload phase0/data/results
./lambda-sync.sh config.sync.env download phase0/data phase0/reports
```

Exclude patterns go in `.syncignore` at the repo root.

> **When running via Claude Code:** See `CLAUDE.md` for the protocol.

## File overview

| File                | Committed to git | Description                           |
|---------------------|------------------|---------------------------------------|
| `lambda-sync.sh`    | Yes              | The sync script                       |
| `sync.env.template` | Yes              | Template for personal config          |
| `*.sync.env`        | No (gitignored)  | Your personal credentials and config  |
| `SYNC.md`           | Yes              | This file                             |
