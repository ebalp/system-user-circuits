"""Bootstrap a Lambda Cloud instance for experiment runs.

Automates:
1. Upload .sync.env and configure GitHub credentials
2. Clone the private repo (using GITHUB_TOKEN from .sync.env)
3. Run a setup script (default: lambda-sync.sh setup)
"""

import logging
import posixpath

from lambda_cloud.ssh import SSHConnection

logger = logging.getLogger(__name__)


def _remote_dir_from_url(repo_url: str) -> str:
    """Derive a remote directory path from a git repo URL.

    E.g. "https://github.com/org/my-repo.git" -> "/home/ubuntu/my-repo"
    """
    name = posixpath.basename(repo_url)
    if name.endswith(".git"):
        name = name[:-4]
    return f"/home/ubuntu/{name}"


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
    repo_url: str,
    branch: str = "main",
    remote_dir: str | None = None,
    setup_script: str = "lambda-sync.sh setup",
    install_claude: bool = True,
) -> None:
    """Bootstrap a Lambda instance: credentials, clone, env, setup.

    Order:
      1. Upload .sync.env and configure GITHUB_TOKEN for git
      2. Clone the (private) repo
      3. Move .sync.env into the repo
      4. Run setup script (git identity, uv, Python env)

    Args:
        ssh: Active SSH connection (SSH must already be reachable).
        env_file_path: Local path to the .sync.env file to upload.
        repo_url: Git repo URL to clone.
        branch: Git branch to check out.
        remote_dir: Remote directory to clone into (derived from repo_url by default).
        setup_script: Script to run after cloning (relative to repo root).
        install_claude: Whether to install Claude Code on the instance.
    """
    if remote_dir is None:
        remote_dir = _remote_dir_from_url(repo_url)

    logger.info("Bootstrapping instance %s", ssh.ip)

    # 1. Upload env and configure GitHub credentials BEFORE cloning
    setup_github_credentials(ssh, env_file_path)

    # 2. Clone repo (skip if already exists)
    ssh.run(
        f"test -d {remote_dir} || git clone -b {branch} {repo_url} {remote_dir}",
        timeout=120,
    )
    logger.info("Repo cloned on %s", ssh.ip)

    # 3. Move .sync.env into the repo
    remote_env_path = f"{remote_dir}/.sync.env"
    ssh.run(f"mv /home/ubuntu/.sync.env {remote_env_path}", timeout=10)

    # 4. Run setup script (git identity, uv install, Python env)
    ssh.run(
        f"cd {remote_dir} && test -f {setup_script.split()[0]} && bash {setup_script} || true",
        timeout=300,
    )

    # 5. Auto-source .sync.env on login (OpenRouter, HF_TOKEN, etc.)
    source_line = f'test -f {remote_env_path} && source {remote_env_path}'
    ssh.run(
        f"grep -qF '.sync.env' ~/.bashrc || echo '{source_line}' >> ~/.bashrc",
        timeout=10,
    )

    # 6. Install Claude Code (auth via OpenRouter if configured in .sync.env)
    if install_claude:
        ssh.run(
            "command -v claude >/dev/null 2>&1 || "
            "{ curl -fsSL https://claude.ai/install.sh | bash && "
            "grep -q 'local/bin' ~/.bashrc || echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> ~/.bashrc; }",
            timeout=120,
        )
    logger.info("Instance %s bootstrapped", ssh.ip)
