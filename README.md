# Vision — Self-hosted AI Video Generation

Mission: fine-tune a pretrained video foundation model with LoRA on the author's RTX PRO
6000 Blackwell server to generate high-quality 9:16 social media advertising clips.

- Master issue: https://github.com/tamerabuhalaweh/Vision/issues/1
- Phase issues: #2 G0 inventory/stop, #3 G1 bootstrap, #4 G2 dataset+baseline, #5 G3 model/smoke,
  #6 G4 HF checkpointing, #7 G5 evaluation/evacuation.
- Primary base-model candidate: LTX-2.5 (Lightricks), official trainer `ltx-trainer`, `i2v_lora` config (trains T2V+I2V).
- Fallback: NVIDIA Cosmos Predict2.5 2B (LoRA), Helios as benchmark reference only.

## Security

Never commit HF tokens, passwords, server credentials, private logs, datasets, model weights,
or generated videos to this repository. See SECURITY.md.

## Model selection (provisional, evidence-gated)

- Primary: **LTX-2.5** (Lightricks) 22B transformer, split-pack checkpoints, official `ltx-trainer`.
  Hardware profile: **80GB+ tier** (96GB VRAM) -> standard `t2v_lora.yaml`-style config (adamw, bf16,
  rank 32, grad checkpointing on). License: LTX-2.x Community License — **no paid license required
  below $10M annual revenue**, which must be confirmed by the owner; adapters are derivatives and
  may only be redistributed under the same license.
- Fallback: NVIDIA Cosmos-Predict2.5-2B LoRA (much smaller; fits mid-class GPUs).
- Helios: benchmark reference for long continuous video, not the training target.

## Layout

- `docs/` runbooks, inventories, decision records, reports
- `configs/` training and inference configuration templates (no secrets)
- `scripts/` checked-in utilities (uploader validation, benchmark harness wrappers)
- `.github/` CI checks that require no GPU and no credentials
