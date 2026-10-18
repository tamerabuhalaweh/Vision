#!/usr/bin/env python3
"""Wan2.2-TI2V-5B downloader (apache-2.0, ungated). Persistent supervised unit target.

Layout:
  /home/administrator/models/wan22        <- TI2V-5B transformer shards
  /data/vision-work/models/wan22          <- umt5-xxl encoder (bf16) + Wan2.2 VAE
"""
import os
import sys
import time
from pathlib import Path

REPO = "Wan-AI/Wan2.2-TI2V-5B"
PLAN = {
    "models_t5_umt5-xxl-enc-bf16.pth": "/data/vision-work/models/wan22",
    "Wan2.2_VAE.pth": "/data/vision-work/models/wan22",
    "diffusion_pytorch_model-00001-of-00003.safetensors": "/home/administrator/models/wan22",
    "diffusion_pytorch_model-00002-of-00003.safetensors": "/home/administrator/models/wan22",
    "diffusion_pytorch_model-00003-of-00003.safetensors": "/home/administrator/models/wan22",
}

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    from huggingface_hub import hf_hub_download

    for name, target in PLAN.items():
        tdir = Path(target)
        tdir.mkdir(parents=True, exist_ok=True)
        local = tdir / name
        if local.is_file() and local.stat().st_size > 0:
            log(f"EXISTS {name} bytes={local.stat().st_size}")

    failures = {}
    remaining = [n for n in PLAN if not (Path(PLAN[n]) / n).is_file()]
    log("PLAN_TOTAL_FILES", len(PLAN), "REMAINING", remaining)

    deadline = time.time() + 60 * 60 * 8
    while remaining and time.time() < deadline:
        name = remaining[0]
        target = PLAN[name]
        t0 = time.time()
        try:
            path = hf_hub_download(repo_id=REPO, filename=name, local_dir=target)
            dur = max(0.1, time.time() - t0)
            size = os.path.getsize(path)
            log(f"DL_OK {name} bytes={size} seconds={dur:.0f} mbytes_per_s={size/dur/1e6:.1f}")
            failures.pop(name, None)
            remaining.pop(0)
        except Exception as e:
            failures[name] = failures.get(name, 0) + 1
            log(f"DL_RETRY n={failures[name]} {name} {type(e).__name__} {str(e)[:160]}")
            time.sleep(5 + min(120, 10 * failures[name]))
            if failures[name] >= 30:
                log("DL_GIVING_UP", name)
                remaining.pop(0)

    if remaining:
        log("WAN_DL_INCOMPLETE", remaining)
        return 1
    log("WAN_DL_ALL_DONE")
    Path("/home/administrator/vision-work").joinpath("dl-wan.done").touch()
    return 0


if __name__ == "__main__":
    sys.exit(main())
