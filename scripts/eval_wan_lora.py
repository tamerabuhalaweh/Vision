#!/usr/bin/env python3
"""Eval smoke: generate 9:16 using the trained LoRA; save /data/vision-work/eval/eval-01.mp4."""
import os
import sys
import time
import torch

from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

CKPT = "/home/administrator/models/wan22"
LORA = os.environ.get(
    "LORA_WEIGHT",
    "/data/vision-work/runs/wan22-5b-lora-r20261017-190059/step-72.safetensors",
)
OUT = os.environ.get("EVAL_OUT", "/data/vision-work/eval/eval-01.mp4")
PROMPT = os.environ.get(
    "EVAL_PROMPT",
    "Vertical 9:16 smartphone shot of a glass of orange juice on a wooden table, "
    "morning sunlight, slow orbiting camera, cinematic color grade, shallow depth of field",
)
SEED = int(os.environ.get("EVAL_SEED", "14"))
N_FRAMES = int(os.environ.get("EVAL_FRAMES", "97"))

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)
print("pipeline ready", flush=True)

t0 = time.time()
pipe.load_lora(pipe.dit, lora_config=LORA, alpha=1.2, verbose=1)
print("lora loaded in %.1f s from %s" % (time.time() - t0, LORA), flush=True)

t0 = time.time()
video = pipe(
    prompt=PROMPT,
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=SEED,
    tiled=True,
    height=704, width=1280,
    num_frames=N_FRAMES,
)
save_video(video, OUT, fps=24, quality=8)
print("EVAL_DONE seconds=%.1f -> %s (bytes=%d)" % (time.time() - t0, OUT, os.path.getsize(OUT)), flush=True)
