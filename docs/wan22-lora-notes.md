# Wan2.2-TI2V-5B LoRA (secondary production path) — Field Notes

Status 2026-10-17 ~17:35 UTC. LTX-2.5 and Cosmos-Predict2.5-2B stay gated (401/403),
so the operational production model is Wan-AI/Wan2.2-TI2V-5B (Apache-2.0).

## Measured facts

- First inference: 704x1280 vertical, 97 frames @24fps, 50 unipc steps — 4.1 s/it ≈ 3.5 min
  sampling wall-clock per clip, ~7-8 min per clip including model load. VRAM peak ~36GB.
- Speed is I/O-light; GPU is idle between steps (model offload enabled by default).
- Full 30-clip dataset generation running under `vision-gen-wan.service` (systemd user unit,
  restart on failure, logs in /home/administrator/vision-work/logs/gen-wan.log).

## Training setup (DiffSynth-Studio)

- Clone: modelscope/DiffSynth-Studio @ 4e3496991455b300fd60ab023113422cd16899f5.
- Existing venv `venv-wan` already has the full dependency stack (torch 2.14.1+cu130,
  transformers, diffusers, imageio, decord...) — verified by `generate.py --help` running clean.
- Script: examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B.sh; uses train.py arguments
  including:
  * --height 480 --width 832 (landscape; flipped to 832x480 vertical for our 9:16 set)
  * --num_frames 49 with time_division_factor 4, remainder 1
  * --lora_rank 32, --lora_target_modules "q,k,v,o,ffn.0,ffn.2"
  * --extra_inputs "input_image" (TI2V: first video frame as the conditioning image)
- Checkpoints: local export under save_steps; `--save_steps` writes step-NNN.safetensors
  with prefix `pipe.dit.` stripped by --remove_prefix_in_ckpt "pipe.dit.".
- Dataset format: metadata.csv with columns `video,prompt` under the dataset_base_path;
  the loader uses `UnifiedDataset` and reads `data_file_keys=video`.
- Provenance record persisted to /data/.../self-gen/manifest.json; unaltered hash list.

## Steps executed on the server

1. Wan-AI/Wan2.2-TI2V-5B components downloaded (2 files + 3 safetensors shards
   ~ 22GB). Files recorded in logs (DL_OK phase).
2. Attention patch applied to `wan/modules/attention.py` (with .orig backup) to fall back
   to `scaled_dot_product_attention` when flash-attn is unavailable. The original
   flash-attn path still asserts completion; the alternative fallback avoids the
   dependency, but the training/inference speed penalty (~2x) versus flash-attn3 is
   acceptable for 2h training windows.
3. Wan-AI/Wan2.2-TI2V-5B generate.py succeeded end-to-end once (see baseline mp4 in
   /data/vision-work/dataset/self-gen/).
