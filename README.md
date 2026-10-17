# Vision — Self-hosted AI Video Generation

Mission: fine-tune a pretrained video foundation model with LoRA on the author's RTX PRO
6000 Blackwell server to generate high-quality 9:16 social media advertising clips.

- Master issue: https://github.com/tamerabuhalaweh/Vision/issues/1
- Phase issues: #2 G0 inventory/stop, #3 G1 bootstrap, #4 G2 dataset+baseline, #5 G3 model/smoke,
  #6 G4 HF checkpointing, #7 G5 evaluation/evacuation.
- Primary base-model candidate: LTX-2.5 (Lightricks), official trainer `t2v_lora` configs.
- Fallback: NVIDIA Cosmos Predict2.5 2B (LoRA), Helios as benchmark reference only.

## Security

Never commit HF tokens, passwords, server credentials, private logs, datasets, model weights,
or generated videos to this repository. See SECURITY.md.

## Layout

- `docs/` runbooks, inventories, decision records, reports
- `configs/` training and inference configuration templates (no secrets)
- `scripts/` checked-in utilities (uploader validation, benchmark harness wrappers)
- `.github/` CI checks that require no GPU and no credentials
