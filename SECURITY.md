# Security Policy

## Data classification under this repository (PUBLIC)

Allowed: sanitized runbooks, inventories with placeholder names, configuration templates,
code without secrets, decision records, benchmark result summaries.

Forbidden (do not commit):
- Hugging Face tokens (`hf_*`), GitHub PATs, server passwords, API keys
- Server IP addresses, hostnames, ports, SSH keys
- Container/env file contents from the GPU server, unit files containing credentials
- Datasets or footage regardless of license status
- Trained weights, adapters, latents or generated mp4s

## Local credential handling contract

- `HF.txt` (Windows: `%USERPROFILE%\Desktop\HF.txt`) is read only by local helper scripts
  into process memory; it must never be echoed, logged, or transferred to any file inside
  this repository or its Git history.
- Server credentials are likewise read from their owner-provided file at runtime only.
- Remote authorization uses ephemeral in-memory injection over SSH stdin; tokens are never
  passed through command-line arguments of long-lived host processes.

## If a secret is ever committed

Rotate immediately (HF token + server password), then purge via history rewrite only with
owner approval (this repository is public; rewriting never deletes anything on HF).
