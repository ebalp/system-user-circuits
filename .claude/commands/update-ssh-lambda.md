---
description: Update the SSH lambda host to a new IP address, updating known_hosts and ~/.ssh/config
argument-hint: <new-ip>
allowed-tools: Bash(ssh-keygen:*), Bash(ssh:*), Read, Edit
---

## Task

Update the SSH `lambda` host configuration to use the new IP address: `$ARGUMENTS`

## Steps

1. Read `~/.ssh/config` to find the current IP for the `lambda` host
2. Remove the old IP from `~/.ssh/known_hosts` using `ssh-keygen -R <old-ip>`
3. Also remove the new IP from `~/.ssh/known_hosts` if present: `ssh-keygen -R $ARGUMENTS`
4. Update `~/.ssh/config` — change the `HostName` under `Host lambda` to `$ARGUMENTS`
5. Test the connection: `ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=accept-new lambda echo "connection ok"`
6. Report whether the connection succeeded or failed
