#!/bin/bash
# Wan2.2-TI2V-5B LoRA TRAINING SMOKE (30-step, small set) — DiffSynth-Studio
set -u
cd /home/administrator/vision-work/DiffSynth-Studio
export PATH="/home/administrator/vision-work/venv-wan/bin:/home/administrator/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
systemctl --user stop vision-gen-wan.service 2>/dev/null || true
ls -la /data/vision-work/dataset/self-gen/*.mp4 2>/dev/null | wc -l

RUN_OUT=/data/vision-work/runs/wan22-5b-lora-smoke-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RUN_OUT"

/home/administrator/vision-work/venv-wan/bin/python /home/administrator/vision-work/scripts/dataset_prep.py /data/vision-work/dataset/self-gen

timeout 1800 accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path /data/vision-work/dataset/self-gen \
  --dataset_metadata_path /data/vision-work/dataset/self-gen/metadata.csv \
  --height 480 --width 832 --num_frames 49 \
  --dataset_repeat 2 \
  --data_file_keys "video" \
  --tokenizer_path /home/administrator/models/wan22/google/umt5-xxl \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$RUN_OUT" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" \
  2>&1 | tail -40

echo "=== run dir ==="
ls -la "$RUN_OUT" 2>/dev/null
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
