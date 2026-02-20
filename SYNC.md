# Lambda Data Sync

Persist experiment data across Lambda AI instances and regions.

**Code goes in GitHub. Data goes in the bucket.**

The bucket stores only outputs that shouldn't be in git but need to persist: experiment results, generated reports, and similar data files. Everything else lives in the repo.

The sync script uploads all `*/data/` and `*/reports/` directories from the repo to the bucket, and downloads them back when you switch instances. Code files are never touched.

## One-time account setup

In the [Lambda Cloud console](https://cloud.lambda.ai):

1. **Create personal filesystems** in each region you plan to use — name them after yourself with the location (e.g., `your-name-fs-dc-2`, `your-name-fs-virginia`). Attach the right one when launching an instance.
2. **Create a personal bucket** under **Filesystem → S3 Adapter Filesystems**. One bucket per person, works from any region.
3. **Get S3 credentials** under **Filesystem → S3 Adapter Keys**. Generate an access key.
4. **Get a GitHub token** under **GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)** with `repo` scope.

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

```bash
git clone https://github.com/ebalp/system-user-circuits.git
cd system-user-circuits
# upload your .sync.env to this directory via Jupyter
./lambda-sync.sh config.sync.env setup     # configure git credentials
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
./lambda-sync.sh <config-file> <mode>
```

| Mode       | What it does                                                |
|------------|-------------------------------------------------------------|
| `setup`    | Configures git identity and GitHub credentials              |
| `upload`   | Syncs all `*/data/` and `*/reports/` dirs to bucket         |
| `download` | Syncs bucket data to local repo                             |

Both `upload` and `download` ask for one confirmation. Read the direction carefully — upload overwrites the bucket, download overwrites local data.

> **When running via Claude Code:** See `CLAUDE.md` for the protocol.

## File overview

| File                | Committed to git | Description                           |
|---------------------|------------------|---------------------------------------|
| `lambda-sync.sh`    | Yes              | The sync script                       |
| `sync.env.template` | Yes              | Template for personal config          |
| `*.sync.env`        | No (gitignored)  | Your personal credentials and config  |
| `SYNC.md`           | Yes              | This file                             |
