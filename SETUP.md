# Setup

## 1. Claude Code

Install Claude Code on your Lambda instance:

```bash
npm install -g @anthropic-ai/claude-code
```

Follow the setup guide at https://code.claude.com/docs/en/setup

The Pro plan ($20/month) is more than enough for our workload.

## 2. Instance bootstrap

Once Claude Code is installed, run it and say:

> Clone https://github.com/ebalp/system-user-circuits and set up the instance

It will walk you through everything. Have your `<your-name>.sync.env` file ready to upload, or your credentials handy (see `SYNC.md`).
