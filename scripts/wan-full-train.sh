#!/bin/bash
# Full Wan2.2-TI2V-5B LoRA training run (30 clips, 60 epochs*repeat -> ~180 steps)
# Save checkpoint every 20 steps; upload every checkpoint (exit-safe) via uploader.
set -u
cd /home/administrator/vision-work/DiffSynth-Studio
export PATH="/home/administrator/vision-work/venv-wan/bin:/home/administrator/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
mkdir -p /data/vision-work/runs
RUN_OUT="/data/vision-work/runs/wan22-5b-lora-r$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_OUT"

/home/administrator/vision-work/venv-wan/bin/python /home/administrator/vision-work/scripts/dataset_prep.py /data/vision-work/dataset/self-gen

timeout 7200 accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path /data/vision-work/dataset/self-gen \
  --dataset_metadata_path /data/vision-work/dataset/self-gen/metadata.csv \
  --height 480 --width 832 --num_frames 49 \
  --dataset_repeat 4 \
  --data_file_keys "video" \
  --tokenizer_path /home/administrator/models/wan22/google/umt5-xxl \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 2 \
  --save_steps 20 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$RUN_OUT" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" \
  > "$RUN_OUT/train.log" 2>&1

# After training completes (or times out), push all available checkpoints.
tail -20 "$RUN_OUT/train.log"
for ck in "$RUN_OUT"/step-*.safetensors "$RUN_OUT"/epoch-*.safetensors; do
  [ -f "$ck" ] || continue
  echo "Registered: $ck ($(stat -c%s "$ck") bytes)"
done

echo "=== uploader dry-run policy ==="
# owner-approved token-dependent LIVE_UPLOAD is triggered separately (see policies).
echo "Run dir: $RUN_OUT"
ls -la "$RUN_OUT" | head -30
