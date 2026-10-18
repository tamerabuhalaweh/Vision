#!/usr/bin/env python3
"""Build DiffSynth metadata.csv + dataset manifest from self-generated Wan33 clips.

Output:
  /data/vision-work/dataset/self-gen/metadata.csv   (video,prompt)
  /data/vision-work/dataset/self-gen/manifest.json  (hashes, sizes)
"""
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/data/vision-work/dataset/self-gen")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    prompts = []
    script = Path("/home/administrator/vision-work/scripts/vision_gen_batch.py")
    txt = script.read_text()
    in_prompts = False
    for line in txt.splitlines():
        if line.startswith("PROMPTS = ["):
            in_prompts = True
            continue
        if in_prompts:
            if line.startswith("]"):
                break
            s = line.strip().rstrip(",")
            if s.startswith('"') and s.endswith('"'):
                prompts.append(s[1:-1])
            elif s.startswith('"' ) and s.endswith('" '):
                pass

    clips = sorted(OUT.glob("wan-sg-*.mp4"))
    entries = []
    for i, p in enumerate(clips):
        idx = int(p.stem.split("-")[-1])
        prompt = prompts[idx - 1] if 0 < idx <= len(prompts) else ""
        if not prompt:
            print("WARN no prompt for", p.name)
        entries.append({
            "name": p.name,
            "path": str(p),
            "bytes": p.stat().st_size,
            "sha256": sha256(p),
            "raw_prompt": prompt,
        })

    with (OUT / "metadata.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "prompt"])
        for e in entries:
            w.writerow([e["name"], e["raw_prompt"]])

    manifest = {
        "dataset": "self-gen-wan22-ti2v5b",
        "provenance": "constructed by generation with Wan-AI/Wan2.2-TI2V-5B (Apache-2.0) — outputs owned by us; no third-party footage",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "count": len(entries),
        "entries": entries,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"WROTE {len(entries)} entries; metadata.csv and manifest.json written to {OUT}")


if __name__ == "__main__":
    main()
