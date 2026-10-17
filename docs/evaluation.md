# Evaluation Plan (G5)

Baseline evaluation reference set: `/data/vision-work/dataset/self-gen/wan-baseline-01.mp4`
(and incremental 704p baseline-only clips as "validation-set 8"; captured BEFORE training and
kept untouched on disk per append-only collection discipline).

## Fixed prompts / fixed seeds policy

| # | Purpose | Prompt | Seed | Size / frames |
|---|---|---|---|---|
| B1 | product macro | glass of orange juice, wooden table, morning light, orbit | 14 | 704×1280, 97f |
| B2 | product dark | matte black earbuds case on marble, cool accent light | 42 | " " |
| B3 | hands product | ceramic matcha bowl on linen cloth, slow motion | 7 | " " |
| B4 | lifestyle seasonal | croissant basket in warm oven glow, close orbit | 21 | " " |

(These will be exactly matched by `num_frames=97, 704x1280, seed=B*` validation runs of the trained pipeline; the
diffSynth inference path is available `wan_generate_video.py --task ...` in musubi or Wan-AI's official generate.py).

## Automated regression metrics over the LoRA-trained model

1. **Identity drift:** track rotations and colors via a fixed weighted perceptual hash on the
   centerframe against the baseline (`imagehash.phash`; ~600ms on the CPU, no GPU cost).
2. **VAE latent maximum activation distance** between the baseline (untrained same clip) and
   trained output (same seed) on the FIRST 5 validation prompts; target = closest to 0 but not 0.
3. **Temporal consistency score:** smoothness of optical flow magnitudes (a stub of the pipeline
   provides no OF; we fall back to frame-difference smoothness measured via `imagehash`).
4. **Human track:** render .mp4 plus static thumbnails, diff-compare crop strip between baseline
   and trained; the owner sees at least 3 juxtaposed frames per validation prompt.

## Eval cadence (LoRA smoke and full run)

- Step-wise: at every checkpoint save, run `wan_generate_video.py` with the saved LORA at the SAME
  settings listed above; log latency/sec/frame and size to GitHub data in `docs/evaluation.md`
  visible checkin.
- NEVER mark the run complete without the three validations per trained checkpoint range
  (`B1`…`B3`).
