# Setup

## 1. Claude Code

Install Claude Code on your Lambda instance:

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

Follow the setup guide at https://code.claude.com/docs/en/setup for authentication.

The Pro plan ($20/month) is more than enough for our workload.

## 2. Shared team bucket

Data is stored in a shared Lambda filesystem in Washington DC (`us-east-2` or `us-east-3`), which exposes an S3-compatible API. There is one per team — ask your team lead for the `.sync.env` credentials file if you don't have it yet.

You do **not** need to attach any filesystem when launching an instance. The local instance disk (512GB) is used as ephemeral working space; all results are pushed to the shared bucket before terminating.

See `SYNC.md` for full details on how the bucket sync works.

## 3. Instance bootstrap

Launch an instance without attaching any filesystem. Once Claude Code is installed, run it and say:

> Clone https://github.com/ebalp/system-user-circuits and set up the instance

It will walk you through everything. Have your `.sync.env` credentials file ready to paste or upload (see `SYNC.md`).
