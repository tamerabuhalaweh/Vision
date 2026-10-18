# GPU Service Stop / Restore Runbook (sanitized)

Parent: issue #2 (G0). Synthetic workload names; exact unit names are known to the
operator in the server-side log but are masked in this public file.

## Scope and policies honored

- STOP means STOP. Nothing was deleted: no `docker prune`, no recursive deletion, no
  `killall`, no forced kills, no service disabling, no boot-preset changes.
- Only the two owner-authorized GPU-heaviest workloads were stopped:
  * unit-canary-app  (vLLM inference canary, 77,952 MiB VRAM)
  * unit-canary-bridge (bash-level bridge for the canary endpoint, 0 MiB VRAM)
- Keeping business services alive: unit-tts-bridge (3,130 MiB), unit-asr-worker (2,530 MiB),
  nvidia metrics exporter, all databases, web stacks, tunnels, monitoring.

## Before-stop snapshot (stored server-side, timestamped log)

- `systemctl is-active` for the 4 related units: all `active`
- `nvidia-smi --query-compute-apps`: 3 processes visible (77,952 / 3,130 / 2,530 MiB)
- GPU 83,631 MiB used / 13,621 MiB free
- Restart policy: Restart=always RestartSec=10 (units stay enabled after stop)

## Stop procedure (executed, verified 2026-10-17)

```
sudo systemctl stop unit-canary-bridge.service unit-canary-app.service
sleep 12; systemctl is-active unit-canary-app unit-canary-bridge   # expect inactive, inactive
nvidia-smi --query-compute-apps=... --format=csv                    # expect only voice services
```
Observation for 25s after stop: no auto-restart (explicit stop suppresses Restart=always).

## Restore procedure (tested logic; execute exact commands)

```
sudo systemctl start unit-canary-app.service unit-canary-bridge.service
systemctl is-active unit-canary-app unit-canary-bridge               # expect active, active
journalctl -u unit-canary-app -n 50                                  # confirm /health 200
nvidia-smi --query-compute-apps=...                                  # EngineCore back at ~78GB
```
Startup time for the canary may take several minutes while loading weights; that is normal.
The unit is `enabled`, so an OS reboot will also auto-restore it.

## Failure/rollback notes

- If the canary fails to restart first try: check journal, then `sudo systemctl start` again
  (Restart=always plus 10s RestartSec already retries transient startup).
- Nothing else on this server depends on GPU for our mission; if voice services are
  inadvertently stopped, restore with
  `sudo systemctl start unit-tts-bridge unit-asr-worker` (unit names per server log).
- If a reboot happens during the mission and the canary auto-returns, re-apply the stop
  procedure (business proxy stacks, DBs and web apps are NOT related to the canary).
