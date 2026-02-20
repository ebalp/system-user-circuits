# Setup

## 1. Claude Code

Install Claude Code on your Lambda instance:

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

Follow the setup guide at https://code.claude.com/docs/en/setup for authentication.

## 2. Personal bucket

Data is stored in a personal Lambda filesystem in Washington DC (`us-east-2` or `us-east-3`), which exposes an S3-compatible API. There is one per person — see `SYNC.md` for how to create yours.

You do **not** need to attach any filesystem when launching an instance. The local instance disk (512GB) is used as ephemeral working space; all results are pushed to the personal bucket before terminating.

See `SYNC.md` for full details on how the bucket sync works.

## 3. Instance bootstrap

Launch an instance without attaching any filesystem. Once Claude Code is installed, run it and say:

> Clone https://github.com/ebalp/system-user-circuits and set up the instance

It will walk you through everything. Have your `.sync.env` credentials file ready to paste or upload (see `SYNC.md`).
