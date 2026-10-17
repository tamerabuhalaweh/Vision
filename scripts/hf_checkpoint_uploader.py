#!/usr/bin/env python3
"""
Append-only Hugging Face checkpoint uploader (see docs/checkpoint-recovery.md).

Usage:
  python hf_checkpoint_uploader.py --repo-id TamerITIS/vision-social-video-YYYYMMDD-abc123 \
      --step-dir /path/to/trainer/checkpoint_dir --step 250 --run-id abc123 \
      [--tag adapter|state] [--dry-run]

Env (required, never stored on disk):
  HUGGING_FACE_HUB_TOKEN  (or HF_TOKEN)

Guarantees:
- Refuses to touch anything if the remote target directory already contains files
  (append-only; a retry uses a new -rN suffix instead of overwriting).
- Uploads a manifest.json with sha256 of every file.
- Verifies after upload (path list + sizes); prints UPLOAD_VERIFIED or FAILED.
- Exits 0 on verified success; 2 fs missing; 3 remote/API error; 4 missing token.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import shutil
from pathlib import Path

RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 20.0
AUDIT_LOG = Path("upload-audit.jsonl")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024.0
    return f"{n:.1f}GB"


def die(code: int, msg: str):
    print(f"FAILED {msg}", flush=True)
    sys.exit(code)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--step-dir", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--tag", default="adapter")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        die(4, "NO_TOKEN_IN_ENV")

    step_dir = Path(args.step_dir).resolve()
    if not step_dir.is_dir():
        die(2, f"STEP_DIR_MISSING {step_dir}")

    files = sorted(
        p for p in step_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {
            ".safetensors", ".pt", ".json", ".yaml", ".yml", ".txt", ".mp4", ".md",
        }
    )
    if not files:
        die(2, f"NO_UPLOADABLE_FILES {step_dir}")

    remote_prefix = f"checkpoints/step-{args.step:07d}-{args.run_id}"
    if args.tag:
        remote_prefix += f"-{args.tag}"

    entries = []
    for p in files:
        entries.append({
            "name": p.name,
            "size": p.stat().st_size,
            "sha256": sha256(p),
        })
    manifest = {
        "run_id": args.run_id,
        "step": args.step,
        "tag": args.tag,
        "repo_id": args.repo_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": entries,
    }

    total = sum(e["size"] for e in entries)
    print(f"PLAN step={args.step} files={len(entries)} bytes={total} ({human(total)})")
    for e in entries:
        print(f"  {e['name']} {e['size']} {e['sha256'][:12]}")
    manifest_path = step_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    entries.append({"name": "manifest.json", "size": manifest_path.stat().st_size,
                    "sha256": sha256(manifest_path)})

    if args.dry_run:
        print(f"DRY_RUN would upload prefix {remote_prefix}")
        return

    from huggingface_hub import HfApi  # deferred import
    api = HfApi(token=token)

    # Append-only guard: refuse if target prefix already populated.
    try:
        existing = api.list_repo_files(repo_id=args.repo_id, repo_type="model")
    except Exception as e:
        die(3, f"LIST_FAILED {type(e).__name__}: {str(e)[:200]}")
    clash = [k for k in existing if k.startswith(remote_prefix + "/")]
    if clash:
        die(3, f"APPEND_ONLY_CLASH prefix exists with {len(clash)} files "
                f"first={clash[0]}")

    # Stage files in a temp dir so remote layout is exactly remote_prefix/<name>.
    with tempfile.TemporaryDirectory(prefix="hfcu.") as td:
        staging = Path(td) / "payload"
        staging.mkdir()
        for e in entries:
            src = manifest_path if e["name"] == "manifest.json" else step_dir / e["name"]
            shutil.copy2(src, staging / e["name"])

        last_err = None
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                t0 = time.time()
                api.upload_folder(
                    folder_path=str(staging),
                    repo_id=args.repo_id,
                    repo_type="model",
                    path_in_repo=remote_prefix,
                    commit_message=f"ckp step-{args.step:07d} run-{args.run_id} ({args.tag})",
                    etag_timeout=60,
                )
                print(f"UPLOAD_OK attempt={attempt} seconds={time.time() - t0:.1f}")
                break
            except Exception as e:
                last_err = e
                print(f"RETRY attempt={attempt} {type(e).__name__}: {str(e)[:200]}")
                time.sleep(RETRY_BACKOFF * attempt)
        else:
            die(3, f"UPLOAD_RETRIES_EXHAUSTED {type(last_err).__name__}")

    # Verification: paths + sizes.
    try:
        infos = api.get_paths_info(repo_id=args.repo_id, repo_type="model",
                                   paths=[f"{remote_prefix}/{e['name']}" for e in entries])
        ok = True
        for e in entries:
            want = e["size"]
            got = None
            for info in infos:
                if getattr(info, "path", None) == f"{remote_prefix}/{e['name']}":
                    got = getattr(info, "lfs", None) or getattr(info, "blob", None)
                    got = getattr(got, "size", None) or getattr(got, "size", None)
                    break
            status = "ok" if (got == want and got is not None) else f"MISMATCH want={want} got={got}"
            if status != "ok":
                ok = False
            print(f"  VERIFY {e['name']} {status}")
        if not ok:
            die(3, "VERIFY_MISMATCH — do NOT trust this checkpoint; use -r2 retry dir")
        print("UPLOAD_VERIFIED")
    except Exception as e:
        die(3, f"VERIFY_ERROR {type(e).__name__}: {str(e)[:200]}")

    rec = {"ts": time.time(), "repo": args.repo_id, "step": args.step, "run": args.run_id,
           "files": [e["name"] for e in entries], "bytes": total, "status": "verified"}
    with AUDIT_LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
