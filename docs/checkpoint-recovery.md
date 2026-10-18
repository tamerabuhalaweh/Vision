# Checkpoint Recovery Plan (G4)

Goal: an never-delete, immutable, verified Off-server checkpoint trail on Hugging Face
(under TamerITIS namespace, private), BEFORE training is left unattended.

## Repository policy

- ONE new private model repo: `TamerITIS/vision-social-video-20261017-<run-id>` (created fresh,
  `exist_ok=False`; unique 6-hex run-id per attempt). NEVER delete or overwrite existing content.
- Append-only storage layout:
  `checkpoints/step-NNNNNNN-<run-id>/lora_weights_step_NNNNNNN.safetensors`
  `checkpoints/step-NNNNNNN-<run-id>/training_state_step_NNNNNNN.pt` (optimizer/scheduler/RNG when supported)
  `checkpoints/step-NNNNNNN-<run-id>/manifest.json` (file list, sizes, sha256, trainer commit, seed, config hash)
  `checkpoints/step-NNNNNNN-<run-id>/spec.txt` (model id/revision, dataset manifest hash, steps, VRAM, timestamp)
- The HF uploader (`scripts/hf_checkpoint_uploader.py`) refuses to write to a path that exists and
  uses create-only repos (`exist_ok=False`). Failed uploads are reported FAILED for owner review,
  never falsely marked successful; retry uses a *new* directory suffix `-r2` instead of overwrite.

## Cadence

- Training starts only AFTER the first smoke-test checkpoint round-trip (upload + list-verify +
  download + hash) is proven on the exact target repo.
- Thereafter: adapter+anchor checkpoint upload ≤ every 30–60 minutes (checkpoint save interval
  250 steps; we will set interval ≤ ADJUSTED for early low-latency backups).
- Full resumable state (optimizer etc.) every ~2–3 hours or when bandwidth allows, to cap loss on
  failure grids.

## Interleaved verification

- After each upload, list repo files and COMPARE bytes/size; for one file per epoch re-hash after
  download. Log `UPLOAD_VERIFIED` lines into issue #6 (fail-safe honesty rule).
- Test restore-from-downloaded checkpoint at least once before unattended training resumes.

## Local/server-side safety

- Local `run-state/` (server) keeps the last two checkpoints + logs. Nothing else is stored
  longer than needed; never pruned dataset or model weights.

## Wake-up-from-failure procedure (resumable)

1. Verify the server process/venv state (`systemctl --user` units or a fresh `uv run python scripts/train.py`).
2. Point `model.load_checkpoint` at the freshest verified checkpoint directory (server local OR
   freshly downloaded from HF private repo).
3. Resume run; verify loss continuity and validation-sample regeneration.
4. Update issue #6 with `RESUMED` + number of the restored checkpoint and its HF URL.

## Token handling on the server (click-safe)

HF token is available ONLY as `HUGGING_FACE_HUB_TOKEN` environment variable injected per-session
in memory (never on disk server-side; never through command-line args). Long-running uploaders
inherit the env when launched by the session controller. If an uploader must be restarted
unattended — acceptable trade-off: the owner periodically re-authorizes token injection.
