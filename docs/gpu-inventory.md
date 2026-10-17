# GPU Inventory (sanitized)

Captured: 2026-10-17 ~14:00 UTC. Source: authorized live SSH probe with owner-provided
credentials handled in memory only. Server identity, IP, and protected product paths are
deliberately omitted from this public document.

## Machine

| Item | Value |
|---|---|
| OS | Ubuntu 24.04 LTS, kernel 6.8.0-139-generic, uptime 27 days at capture |
| GPU | 1x NVIDIA RTX PRO 6000 Blackwell Server Edition (**96 GB-class**, NOT RTX 6000 Ada 48GB) |
| GPU VRAM | 97,887 MiB total |
| Driver / CUDA driver API | 595.58.03 / 13.2 |
| Compute capability | (12, 0) = sm_120 Blackwell |
| CPU | 32 vCPU (Skylake-class virtualized, 1 thread/core) |
| RAM | 82 GiB total, ~70 GiB available at capture; no swap |
| Disk | root 398 GiB (70 GiB free), secondary 492 GiB (43 GiB free); inodes healthy |
| Existing PyTorch | venv A: torch 2.11.0+cu130 (proven running on sm_120); venv B: 2.11.0+cu128 |
| Python (system) | 3.12.3 |
| Key constraint | Persistent storage is tight; only ~113 GiB free across both mounts |

## GPU workloads found (owning account = the server's single admin user)

| Synthetic name | What it is | GPU mem | Managed by | State at capture |
|---|---|---|---|---|
| unit-canary-app | vLLM OpenAI-API inference server, model `nvidia/Gemma-4-26B-A4B-NVFP4` (26B MoE NVFP4), loopback-bound, max-model-len 65536, gpu-mem-util 0.80 | 77,952 MiB | systemd, enabled, Restart=always(10s) | running since Sep 30; journal shows near-idle (health checks + occasional small requests, last real traffic Oct 12) |
| unit-canary-bridge | Local HTTP bridge exposing the canary through a SmartLLM-compatible endpoint (no GPU use) | 0 | systemd, enabled | running |
| unit-tts-bridge | OmniVoice TTS production bridge | 3,130 MiB | systemd, enabled | running — left untouched to preserve live voice features |
| unit-asr-worker | Whisper-based ASR API worker (child process of STT unit) | 2,530 MiB | systemd, enabled | running — left untouched to preserve live voice features |
| nvidia-exporter | GPU metrics exporter | 0 | systemd | running — left untouched |

## Non-GPU services inventoried and explicitly NOT touched

Docker daemon with 40+ containers (multi Postgres/pgvector/Supabase stacks, n8n+worker+redis,
FreeSWITCH audio stage, node services), nginx, PostgreSQL/Timescale on host, postgrest,
kubectl-free plain processes, cron, fail2ban, WireGuard tunnel interface. No containers,
images, volumes, files, or datasets were pruned, deleted, or modified. Docker restart
policies and service definitions left intact.

## Action taken under owner authorization (graceful STOP, no deletion)

Stopped ONLY: `unit-canary-app.service` and `unit-canary-bridge.service` via
`sudo systemctl stop ...` (systemd honors explicit stop even with Restart=always).

Verified before stopping:
- both units active, log snapshot saved to a server-side timestamped log
- restart policy = `always`/10s, units remain `enabled` (auto-start at reboot preserved)
- app offline drop-in already sets HF_HUB_OFFLINE=1 (restore needs no network change)

Verified after stopping (25s observation window):
- both units `inactive`, NOT restarted by watchdog or timer
- GPU: 83,631 -> 56,743 MiB used; **91,579 MiB free**; utilization 0%
- the two legal voice services remain (5,674 MiB)

## Risk notes

- If the host reboots mid-mission, unit-canary-app is `enabled` and will reclaim ~78GB
  VRAM; re-run the stop commands after any reboot (see stop-restore runbook).
- Disk space is the hardest constraint for hosting LTX-2.5 base weights (~42GB bf16 +
  15-26GB text encoder); plan cache placement carefully before bulk download.
