# Lambda Data Sync

Persist experiment data across Lambda AI instances and regions.

**Code goes in GitHub. Data goes in the bucket.**

The bucket stores only outputs that shouldn't be in git but need to persist: experiment results, generated reports, and similar data files. Everything else lives in the repo.

## How it works

Each team member has:

- **A personal filesystem per region** — named after yourself (e.g., `your-name-fs-virginia`). Lambda mounts it at `/lambda/nfs/<name>` on every instance in that region.
- **A personal bucket on us-east-2** — a single store in Washington DC for your data. When you move between regions, upload before shutting down and download on the new instance.

The bucket uses Lambda's Filesystem S3 Adapter — not AWS, Lambda's own storage with an S3-compatible API.

The sync script uploads all `*/data/` and `*/reports/` directories from the repo to the bucket, and downloads them back when you switch instances. Code files are never touched.

## One-time account setup

### 1. Create your personal filesystems

In the [Lambda Cloud console](https://cloud.lambda.ai), create a filesystem in each region you plan to use. Name them after yourself with the geographic location:

- `your-name-fs-dc-2` (us-east-2, Washington DC — also where your bucket lives)
- `your-name-fs-virginia` (us-east-3)
- `your-name-fs-ohio` (us-midwest-2)
- etc.

When launching instances, attach your filesystem for that region.

### 2. Create your personal bucket

In the Lambda Cloud console, go to **Filesystem → S3 Adapter Filesystems** and create a bucket. One bucket per person is enough — it works from any region.

### 3. Get your S3 Adapter credentials

Go to **Filesystem → S3 Adapter Keys** and generate an access key. These keys work for any bucket.

### 4. Create a GitHub Personal Access Token

Go to **GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)** and generate a token with the `repo` scope.

### 5. Create your config file

```bash
cp sync.env.template <your-name>.sync.env
```

Fill in your details. This file is gitignored — keep it in your repo directory and upload it to new instances manually (e.g. via Jupyter).

### 6. Seed your bucket

```bash
./lambda-sync.sh <your-name>.sync.env setup
./lambda-sync.sh <your-name>.sync.env upload
```

## Daily workflow

### Starting work on a new instance

```bash
./lambda-sync.sh <your-name>.sync.env setup      # configure git credentials
./lambda-sync.sh <your-name>.sync.env download    # restore data from bucket
git pull                                           # get latest code
```

### Before shutting down

```bash
./lambda-sync.sh <your-name>.sync.env upload
```

### Using your HF token

After setup, source your config to make `HF_API_KEY` available in your shell:

```bash
source <your-name>.sync.env
```

### Committing code

`git add`, `git commit`, `git push` as usual. The sync script only touches data directories.

## Script reference

```
./lambda-sync.sh <config-file> <mode>
```

| Mode       | What it does                                                |
|------------|-------------------------------------------------------------|
| `setup`    | Configures git identity and GitHub credentials              |
| `upload`   | Syncs all `*/data/` and `*/reports/` dirs to bucket         |
| `download` | Syncs bucket data to local repo                             |

Both `upload` and `download` show what will happen and ask for one confirmation. Read the direction carefully — upload overwrites the bucket, download overwrites local data.

> **When running via Claude Code:** See `CLAUDE.md` for the protocol.

## Available regions and endpoints

| Region        | Location       | Endpoint                              |
|---------------|----------------|---------------------------------------|
| us-east-2     | Washington DC  | https://files.us-east-2.lambda.ai    |
| us-east-3     | Washington DC  | https://files.us-east-3.lambda.ai    |
| us-midwest-2  | Ohio           | https://files.us-midwest-2.lambda.ai  |

Buckets are created in `us-east-2` but accessible from any region.

## File overview

| File                   | Committed to git | Description                           |
|------------------------|------------------|---------------------------------------|
| `lambda-sync.sh`       | Yes              | The sync script                       |
| `sync.env.template`    | Yes              | Template for personal config          |
| `<your-name>.sync.env` | No (gitignored)  | Your personal credentials and config  |
| `SYNC.md`              | Yes              | This file                             |
