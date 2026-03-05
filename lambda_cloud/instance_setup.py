"""Bootstrap a Lambda Cloud instance for experiment runs.

Automates the manual steps from CLAUDE_LAMBDA.md:
1. Upload .sync.env and configure GitHub credentials
2. Clone the private repo (using GITHUB_TOKEN from .sync.env)
3. Run lambda-sync.sh setup (git identity, uv, Python env)
"""

import logging

from lambda_cloud.ssh import SSHConnection

logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/ebalp/system-user-circuits.git"
REMOTE_REPO_DIR = "/home/ubuntu/system-user-circuits"


def setup_github_credentials(ssh: SSHConnection, env_file_path: str) -> None:
    """Upload .sync.env and configure git to use GITHUB_TOKEN for cloning.

    Must run before git clone so that private repos are accessible.

    Args:
        ssh: Active SSH connection.
        env_file_path: Local path to the .sync.env file.
    """
    # Upload env file to a temp location (repo dir doesn't exist yet)
    remote_tmp_env = "/home/ubuntu/.sync.env"
    ssh.upload_file(env_file_path, remote_tmp_env)
    logger.info("Uploaded .sync.env to %s", ssh.ip)

    # Source the env file and configure git credential rewrite
    ssh.run(
        f'bash -c \'source {remote_tmp_env} && '
        f'if [ -n "$GITHUB_TOKEN" ]; then '
        f'git config --global url."https://${{GITHUB_TOKEN}}@github.com/".insteadOf "https://github.com/"; '
        f'echo "GitHub credentials configured"; '
        f'else echo "WARNING: GITHUB_TOKEN not set in .sync.env"; fi\'',
        timeout=15,
    )
    logger.info("GitHub credentials configured on %s", ssh.ip)


def bootstrap_instance(
    ssh: SSHConnection,
    env_file_path: str,
    repo_url: str = REPO_URL,
    branch: str = "main",
) -> None:
    """Bootstrap a Lambda instance: credentials, clone, env, setup.

    Order:
      1. Upload .sync.env and configure GITHUB_TOKEN for git
      2. Clone the (private) repo
      3. Move .sync.env into the repo
      4. Run lambda-sync.sh setup (git identity, uv, Python env)

    Args:
        ssh: Active SSH connection (SSH must already be reachable).
        env_file_path: Local path to the .sync.env file to upload.
        repo_url: Git repo URL to clone.
        branch: Git branch to check out.
    """
    logger.info("Bootstrapping instance %s", ssh.ip)

    # 1. Upload env and configure GitHub credentials BEFORE cloning
    setup_github_credentials(ssh, env_file_path)

    # 2. Clone repo (skip if already exists)
    ssh.run(
        f"test -d {REMOTE_REPO_DIR} || git clone -b {branch} {repo_url} {REMOTE_REPO_DIR}",
        timeout=120,
    )
    logger.info("Repo cloned on %s", ssh.ip)

    # 3. Move .sync.env into the repo
    remote_env_path = f"{REMOTE_REPO_DIR}/.sync.env"
    ssh.run(f"mv /home/ubuntu/.sync.env {remote_env_path}", timeout=10)

    # 4. Run lambda-sync.sh setup (git identity, uv install, Python env)
    ssh.run(
        f"cd {REMOTE_REPO_DIR} && test -f lambda-sync.sh && bash lambda-sync.sh setup || true",
        timeout=300,
    )

    # 5. Auto-source .sync.env on login (OpenRouter, HF_TOKEN, etc.)
    source_line = f'test -f {remote_env_path} && source {remote_env_path}'
    ssh.run(
        f"grep -qF '.sync.env' ~/.bashrc || echo '{source_line}' >> ~/.bashrc",
        timeout=10,
    )

    # 6. Install Claude Code (auth via OpenRouter if configured in .sync.env)
    ssh.run(
        "command -v claude >/dev/null 2>&1 || "
        "{ curl -fsSL https://claude.ai/install.sh | bash && "
        "grep -q 'local/bin' ~/.bashrc || echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> ~/.bashrc; }",
        timeout=120,
    )
    logger.info("Instance %s bootstrapped", ssh.ip)
