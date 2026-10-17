# Model Selection — Decision Record (G3, provisional)

Date: 2026-10-17. Status: PROVISIONAL until first smoke test + owner gate review.

## Decision

Fine-tune **LTX-2.5** (Lightricks) with LoRA via the official `ltx-trainer` (LTX-2 monorepo,
`packages/ltx-trainer/`). Primary training config: **`i2v_lora.yaml`** (first-frame conditioning at
probability 0.5 → learns I2V *and* T2V at once — required for social ads where generation may start
from a product image or from text). Fallback if infeasible: Cosmos Predict2.5 2B. Long-duration
Helios: benchmark reference only, not the training model.

## Evidence

### Hardware fit (measured)
- GPU: RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, sm_120, driver 595.58.03/CUDA 13.2.
- Official trainer docs: 80GB recommended, 32GB minimum -> our card is the **80GB+ tier** →
  standard config verbatim: `optimizer adamw`, `quantization: null`, `bf16`, batch 1,
  grad-checkpointing on, LoRA rank 32 / alpha 32 (`hardware-profiles.md`).
- Existing venv on the server already proved torch runs on this card (vLLM Gemma-4 NVFP4),
  and our trainer venv installs torch 2.13.0+cu132 with (12,0) device detection verified.

### Components (official split-pack, Hugging Face Lightricks/LTX-2.5)
- `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` (42,018,190,584 B,
  sha256 31eb3cad...) — distilled 22B transformer for fast inference + LoRA fine-tune.
- `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` (26,263,858,182 B) — the LTX
  fine-tuned Gemma4-12b encoder with projection bundled; official docs explicitly warn NOT to
  substitute Google's stock Gemma 4 (checkpoint-compatibility check `gemma4-12b-ltx-v1`).
- `vae/ltx-2.5-video-vae-bf16.safetensors` (1,472,223,346 B) + `vae/ltx-2.5-audio-vae-bf16.safetensors`
  (364,866,540 B) — split-pack required VAEs.
- `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` (995,778,752 B).
- Mixed incompatible checkpoints/components will NOT be used. The dev (+42GB) transformer is NOT
  downloaded (we do not need the guided two-stage dev pipelines; same LoRA, different backbone —
  the distilled LoRA path is the officially supported fast path with `DistilledPipeline`/`DFR`).

### Official trainer facts (from `docs/quick-start.md`, `hardware-profiles.md`, shipped configs)
- Split-pack config requires naming transformer + encoder + both VAEs explicitly.
- Defaults: 2000 steps, lr 1e-4 linear, bf16, adamw, grad ckpt on, keep_all checkpoints every
  250 steps, validation every 100 steps at `video_dims [960,544,89]` (we will patch to vertical
  9:16: `video_dims [544, 960, F]` with F % 8 == 1, e.g. 49 → ~2s validation sample).
- Preprocessing: `process_dataset.py dataset.json --resolution-buckets "544x960x49" ...` computes
  latents + text embeddings offline; caches are checkpoint/encoder-pair specific.
- Captioning can be self-hosted: `scripts/serve_captioner.py` (Qwen3-Omni-30B-A3B, vLLM, FP8 ~31GB
  → fits our 96GB card; no external API keys needed).
- Attention backend: PyTorch SDPA default; FA4 optional on Blackwell (datacenter B200 verified
  upstream on torch 2.9.1+cu128 — we stay on SDPA for the smoke test).

### License findings (LTX-2.x Community License, Aug 2026)
- Entities with ≥$10M annual consolidated revenue must obtain a paid license for any use
  (fine-tuning included). Below that threshold, commercial use under community terms is allowed.
- LoRA adapters are "Derivatives"; distributing them to third parties requires the same license +
  notice; commercial transferees need their own paid license. Private backup storage of our own
  adapter on Hugging Face under the TamerITIS namespace is consistent with these terms.
- Acceptable-use policy governs outputs. Data used in training is NOT licensed under this
  agreement (our data rights must be cleared separately — see dataset-provenance.md).
- **OWNER ACTION REQUIRED:** confirm annual revenue < $10M; accept the LTX-2.5 gate on HF
  ("Agree and Access") with the TamerITIS token user; provide a write-scoped token.

### Fallback path (if LTX-2.5 blocked/bogged under our compute or license)
- NVIDIA Cosmos-Predict2.5-2B (LoRA-capable, smaller; NVIDIA Open Model License — commercial
  terms must still be verified before commit). Helios only as benchmark for long generation.

## Time-plan overlay (48h mission)

| Window | Activity |
|---|---|
| 0–4h | Inventory + stop/restore (DONE 2026-10-17 ~14:00 UTC) + env/bootstrap |
| 4–10h | Baseline inference benchmarks with frozen model (9:16, seeds) |
| 10–14h | Dataset prep for owned/licensed footage; preprocess latents/cache |
| 14–38h | LoRA training + 30–60min checkpoint cadence + verified HF uploads |
| 38–44h | Evaluation vs baseline; 15–30s multi-shot assembly |
| last 4h | Final evacuation and recovery documentation |

Actual allocation expiry and the exact remaining window must be confirmed by the owner.
