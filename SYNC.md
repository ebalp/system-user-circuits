# Lambda Filesystem Sync

Sync your local working data across Lambda AI instances, regardless of region.

Code goes in GitHub. Data (numpy arrays, experiment results, checkpoints, etc.) goes here.

## How it works

Each team member has:

- **A personal filesystem per region** — named after yourself (e.g., `fs-enrique`, `fs-alice`). Lambda mounts it at `/lambda/nfs/<name>` on every instance in that region.
- **A personal bucket on us-east-2** — a single persistent store in Washington DC that holds your data. When you move between regions, you upload before shutting down and download on the new instance.

The bucket uses Lambda's Filesystem S3 Adapter — it's not AWS, it's Lambda's own storage accessed via an S3-compatible API.

## One-time account setup

You only do this once, ever.

### 1. Create your personal filesystems

In the [Lambda Cloud console](https://cloud.lambda.ai), create a filesystem in each region you plan to use. Name them after yourself:

- `fs-enrique` (us-east-2)
- `fs-enrique` (us-midwest-2)
- etc.

When launching instances, attach your filesystem for that region.

### 2. Create your personal bucket

In the Lambda Cloud console, go to **Filesystem → S3 Adapter Filesystems** and create a bucket. This is your persistent store — all your data syncs to and from here. One bucket per person is enough.

The bucket is created in `us-east-2` (Washington DC). It works from any region.

### 3. Get your S3 Adapter credentials

In the Lambda Cloud console, go to **Filesystem → S3 Adapter Keys** and generate an access key. Download the credentials file. These keys are tied to your Lambda account and work for any bucket.

### 4. Create a GitHub Personal Access Token

Go to **GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)** and generate a token with the `repo` scope. This allows pushing and pulling code from new instances.

### 5. Create your config file

```bash
cp sync.env.template <your-name>.sync.env
```

Fill in your details:

```
BUCKET_NAME=<your-bucket-uuid>

LAMBDA_ACCESS_KEY_ID=<from credentials file>
LAMBDA_SECRET_ACCESS_KEY=<from credentials file>
LAMBDA_REGION=us-east-2
LAMBDA_ENDPOINT_URL=https://files.us-east-2.lambda.ai

GIT_USER_NAME="Your Full Name"
GIT_USER_EMAIL=your-email@example.com

GITHUB_TOKEN=<your GitHub classic token>
```

This file is gitignored — it stays in your filesystem, syncs with your bucket, and never goes to GitHub.

### 6. Seed your bucket

```bash
./lambda-sync.sh <your-name>.sync.env setup
./lambda-sync.sh <your-name>.sync.env upload
```

You're done. Your data is now in the bucket.

## Daily workflow

### Starting work on an instance with a fresh/stale filesystem

Your filesystem might be empty or outdated (e.g., you worked in another region and synced there). Pull the latest from your bucket:

```bash
./lambda-sync.sh <your-name>.sync.env setup       # configure git credentials
./lambda-sync.sh <your-name>.sync.env download     # restore your data from bucket
```

The `download` command will warn you that local changes will be overwritten.

### Starting work on an instance where the filesystem is current

If you're resuming on the same instance or the filesystem was already up to date, just set up git:

```bash
./lambda-sync.sh <your-name>.sync.env setup
```

### Before shutting down an instance

Save your data to the bucket:

```bash
./lambda-sync.sh <your-name>.sync.env upload
```

The script will show you the current bucket contents and ask for confirmation before syncing.

### Committing code

Code changes go to GitHub as usual — `git add`, `git commit`, `git push`. The sync script handles data only. Don't put large data files in git.

## Script reference

```
./lambda-sync.sh <config-file> <mode>
```

| Mode       | What it does                                                       |
|------------|--------------------------------------------------------------------|
| `setup`    | Configures git identity and GitHub credentials from your config    |
| `upload`   | Syncs your entire filesystem to your bucket                        |
| `download` | Syncs your bucket to your local filesystem (warns about overwrites)|

The script auto-detects which filesystem you're on from the current directory. Run it from anywhere inside your filesystem mount.

Both `upload` and `download` show what will happen and ask for one confirmation before syncing. Read the direction carefully — upload overwrites the bucket, download overwrites local files.

> **When running via Claude Code:** See `CLAUDE.md` for the protocol.

## Available regions and endpoints

| Region        | Location       | Endpoint                          |
|---------------|----------------|-----------------------------------|
| us-east-2     | Washington DC  | https://files.us-east-2.lambda.ai |
| us-east-3     | Washington DC  | https://files.us-east-3.lambda.ai |
| us-midwest-2  | Ohio           | https://files.us-midwest-2.lambda.ai |

Buckets are created in `us-east-2` but accessible from any region.

## File overview

| File                         | Committed to git | Description                              |
|------------------------------|------------------|------------------------------------------|
| `lambda-sync.sh`             | Yes              | The sync script                          |
| `sync.env.template`          | Yes              | Template for personal config             |
| `<your-name>.sync.env`       | No (gitignored)  | Your personal credentials and config     |
| `SYNC.md`                    | Yes              | This file                                |
