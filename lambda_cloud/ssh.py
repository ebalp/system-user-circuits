"""SSH connection management for Lambda Cloud instances."""

import logging
import subprocess
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Common SSH options to avoid interactive prompts
_SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
]


@dataclass
class SSHConnection:
    """SSH connection to a Lambda Cloud instance.

    All operations go through subprocess SSH calls — no paramiko dependency.
    """
    ip: str
    key_file: str = "~/.ssh/anusha-cre-lambda-key.pem"
    user: str = "ubuntu"

    def _ssh_base(self) -> list[str]:
        """Base SSH command with key and options."""
        return ["ssh", "-i", self.key_file, *_SSH_OPTS]

    def _target(self) -> str:
        return f"{self.user}@{self.ip}"

    def run(self, command: str, timeout: int = 300, check: bool = True) -> subprocess.CompletedProcess:
        """Run a command on the remote instance via SSH.

        Args:
            command: Shell command to execute remotely.
            timeout: Seconds before killing the SSH process.
            check: If True, raise CalledProcessError on non-zero exit.

        Returns:
            CompletedProcess with stdout/stderr captured as strings.
        """
        cmd = [*self._ssh_base(), self._target(), command]
        logger.debug("SSH run: %s", command)
        return subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, check=check,
        )

    def run_background(self, command: str) -> None:
        """Run a command in the background on the remote instance.

        Uses nohup + disown so the process survives SSH disconnection.
        """
        bg_command = f"nohup {command} > /dev/null 2>&1 & disown"
        cmd = [*self._ssh_base(), self._target(), bg_command]
        logger.debug("SSH background: %s", command)
        subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)

    def open_tunnel(self, local_port: int, remote_port: int) -> subprocess.Popen:
        """Open an SSH tunnel (local port forwarding).

        Returns a Popen handle — caller is responsible for terminating it.

        Args:
            local_port: Local port to bind.
            remote_port: Remote port to forward to.

        Returns:
            subprocess.Popen running the SSH tunnel.
        """
        cmd = [
            *self._ssh_base(),
            "-N", "-L", f"{local_port}:localhost:{remote_port}",
            self._target(),
        ]
        logger.info("Opening SSH tunnel localhost:%d -> %s:%d", local_port, self.ip, remote_port)
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def wait_for_ssh(self, timeout: int = 300, interval: int = 10) -> bool:
        """Wait until SSH is reachable on the instance.

        Args:
            timeout: Max seconds to wait.
            interval: Seconds between attempts.

        Returns:
            True if SSH became reachable, False if timed out.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                result = self.run("echo ok", timeout=15, check=False)
                if result.returncode == 0:
                    logger.info("SSH reachable on %s", self.ip)
                    return True
            except (subprocess.TimeoutExpired, OSError) as e:
                logger.debug("SSH not ready on %s: %s", self.ip, e)
            time.sleep(interval)
        logger.warning("SSH not reachable on %s after %ds", self.ip, timeout)
        return False

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a file to the remote instance via SCP.

        Args:
            local_path: Local file path.
            remote_path: Destination path on the remote instance.
        """
        cmd = [
            "scp", "-i", self.key_file, *_SSH_OPTS,
            local_path, f"{self._target()}:{remote_path}",
        ]
        logger.info("SCP %s -> %s:%s", local_path, self.ip, remote_path)
        subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
