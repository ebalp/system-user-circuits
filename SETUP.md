# Setup

## 1. Claude Code

Install Claude Code on your Lambda instance:

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

Follow the setup guide at https://code.claude.com/docs/en/setup for authentication.

The Pro plan ($20/month) is more than enough for our workload.

## 2. Personal filesystems

Before launching an instance, make sure you have two personal filesystems created in the [Lambda Cloud console](https://cloud.lambda.ai):

- **A filesystem in `us-east-2` or `us-east-3` (Washington DC)** — this is where your personal bucket lives. It is your persistent data store across regions.
- **A filesystem in the region where you are launching the instance** — this is what gets mounted on the instance.

Name both filesystems after yourself using the geographic location (e.g., `your-name-fs-dc-2` for the Washington DC bucket filesystem, `your-name-fs-virginia` or `your-name-fs-ohio` for compute instance filesystems). When launching an instance, attach the filesystem for that region.

See `SYNC.md` for full details on how filesystems and buckets work together.

## 3. Instance bootstrap

Once Claude Code is installed, run it and say:

> Clone https://github.com/ebalp/system-user-circuits and set up the instance

It will walk you through everything. Have your `<your-name>.sync.env` file ready to upload, or your credentials handy (see `SYNC.md`).
