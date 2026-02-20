# Setup

## 1. Personal bucket

Data is stored in a personal Lambda filesystem in Washington DC (`us-east-2` or `us-east-3`), which exposes an S3-compatible API. There is one per person — see `SYNC.md` for how to create yours.

## 2. Launch an instance

Launch an instance without attaching any filesystem. The local instance disk (512GB) is used as ephemeral working space; all results are pushed to the personal bucket before terminating.

## 3. Open a terminal

Log into JupyterLab on your Lambda instance and open a terminal from the Launcher (**+** → **Terminal**).

## 4. Claude Code

Install Claude Code on your Lambda instance:

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

Follow the setup guide at https://code.claude.com/docs/en/setup for authentication. Claude Code is also compatible with OpenRouter if you prefer to use that instead.

## 5. Instance bootstrap

Once Claude Code is installed, run it and say:

> Clone https://github.com/ebalp/system-user-circuits and set up the instance

It will walk you through everything. Have your `.sync.env` credentials file ready to paste or upload via Jupyter (see `SYNC.md`).
