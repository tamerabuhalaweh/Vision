#!/usr/bin/env python3
"""Batch vertical 9:16 generation with Wan2.2-TI2V-5B (Apache-2.0 base; we own all outputs).

Writes clips to /data/vision-work/dataset/self-gen/ with 1-based numbering ng-XX.mp4
Skips files that already exist. Designed to run under a systemd user unit.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WAN_ROOT = "/home/administrator/vision-work/Wan2.2"
PY = "/home/administrator/vision-work/venv-wan/bin/python"
CKPT = "/home/administrator/models/wan22"
OUT = Path("/data/vision-work/dataset/self-gen")

PROMPTS = [  # 9:16 social-ad territory; product/lifestyle/hands; varied looks
    "Vertical 9:16 smartphone shot of a glass of orange juice on a wooden table, morning sunlight, slow orbiting camera, cinematic color grade, shallow depth of field",
    "Vertical 9:16 slow push-in on a matte black wireless earbuds case on a marble countertop, cool blue accent light, shallow depth of field",
    "Vertical 9:16 a hand slowly places a ceramic matcha bowl on a linen tablecloth, soft daylight, slow motion, minimal composition",
    "Vertical 9:16 macro shot of water droplets rolling off a fresh green apple, dark background, dramatic rim light",
    "Vertical 9:16 a barista pours latte art into a tall glass, warm cafe ambience, handheld camera, vertical framing",
    "Vertical 9:16 a pair of running shoes on a wet city sidewalk at golden hour, low angle, slow dolly forward",
    "Vertical 9:16 glossy red lipstick tube rotating on a reflective surface, black backdrop, studio spotlight, product commercial",
    "Vertical 9:16 a chef drizzles olive oil onto a tomato basil salad, close-up, natural kitchen window light",
    "Vertical 9:16 a smartwatch showing sunrise alarm on a bedside nightstand, soft fabric tones, slow pan",
    "Vertical 9:16 two hands unfolding a cardboard gift box revealing pastel paper, gentle warm light, overhead shot",
    "Vertical 9:16 ice cubes falling into a sparkling glass of cola with condensation, black background, high speed look",
    "Vertical 9:16 a runner ties neon yellow shoelaces on fresh sneakers, urban park morning, side tracking shot",
    "Vertical 9:16 cosmetic cream jar with swirling texture on a stone surface, soft shadows, top light, minimal aesthetic",
    "Vertical 9:16 a turning key handmade leather wallet on olive fabric, close up stitches, warm analog film look",
    "Vertical 9:16 steam rising from hot cocoa in a ceramic mug near a rainy window, cozy mood, slow tilt",
    "Vertical 9:16 electric skateboard rolling down a smooth boardwalk at sunset, follow shot, lens flare",
    "Vertical 9:16 a person-polishing a vintage brass camera lens with a velvet cloth, moody warm lamp light",
    "Vertical 9:16 lavender soap bars stacked in a spa bathroom, marble and eucalyptus, slow glide camera, misty air",
    "Vertical 9:16 cracked ice cold energy drink can on fitness gym bench, dramatic gym lighting, hand reaches in",
    "Vertical 9:16 a laptop on white desk with potted cactus, screen displays charts, morning light, subtle zoom",
    "Vertical 9:16 a knight chess piece falling in slow motion onto a wooden board, black and white backdrop, dramatic",
    "Vertical 9:16 freshly baked croissants puffed in a rustic basket with flour dusting, warm oven glow, close orbit",
    "Vertical 9:16 colorful sneakers exploding powder cloud against teal wall, freeze-frame feel, studio product shot",
    "Vertical 9:16 a dog's nose sniffs a bowl of fresh pet food, soft home interior, natural light, cute handheld",
    "Vertical 9:16 a woman applies blush with a soft brush in front of a mirror, pink pastel tones, beauty commercial",
    "Vertical 9:16 denim jacket tumbling in slow motion against sky, sunlight through fabric, hyper slow motion look",
    "Vertical 9:16 a barista preparing ps frothy matcha with bamboo whisk, deep green tones, vertical cafe scene",
    "Vertical 9:16 mirror reflection of a coffee cup on a movie-set table, film production mood, anamorphic feel",
    "Vertical 9:16 a row of colorful macarons on slate plate, camera slides right, pastel bakery aesthetic",
    "Vertical 9:16 sunset beam lights up a bottle of amber perfume on a windowsill, particle dust, dreamy slow pan",
]

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    total = len(PROMPTS)
    log("BATCH_TOTAL", total)
    for i, prompt in enumerate(PROMPTS, start=1):
        outp = OUT / f"wan-sg-{i:03d}.mp4"
        if outp.is_file() and outp.stat().st_size > 100_000:
            log(f"SKIP {i} exists")
            continue
        t0 = time.time()
        cmd = [
            PY, str(Path(WAN_ROOT) / "generate.py"),
            "--task", "ti2v-5B", "--size", "704*1280", "--frame_num", "97",
            "--ckpt_dir", CKPT, "--prompt", prompt,
            "--save_file", str(outp),
        ]
        log(f"GEN_START {i}/{total}")
        return_val = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if return_val.returncode != 0 or not outp.exists():
            tail = (return_val.stderr or return_val.stdout or "")[-400:]
            log(f"GEN_FAIL {i} rc={return_val.returncode} tail={tail!r}")
            continue
        log(f"GEN_OK {i} bytes={outp.stat().st_size} seconds={time.time()-t0:.0f}")
    log("BATCH_DONE")

if __name__ == "__main__":
    main()
