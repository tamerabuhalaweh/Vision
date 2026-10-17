#!/usr/bin/env python3
"""Append-only HF checkpoint uploader — see docs/checkpoint-recovery.md.

Revised per issue #9:
- --dry-run is fully OFFLINE and runs before any token check or hub import.
- Single atomic upload via HfApi.create_commit with parent_commit optimistic
  concurrency guard; NO auto-retry of an uncertain commit (operator reinstates
  with a fresh unique prefix; nothing is ever overwritten or deleted).
- Files referenced directly by CommitOperationAdd path (no staging copies);
  manifest.json is generated in a temp dir outside the source checkpoint dir.
- Remote verification: info.size (fallback info.lfs.size), treat size=0 valid,
  validate LFS sha256 when present, and download+re-hash manifest.json.
- Fail closed on: missing/invalid token, non-private repo, existing remote
  prefix, symlinked entries, unapproved file suffixes, failed verification.
- Error text is sanitized (no tokens, truncated); audit log stored outside the
  source repo (VISION_AUDIT_DIR or $HOME/vision-audit/upload-audit.jsonl).

Exit codes: 0 verified; 2 local/fs; 3 remote/uncertain; 4 missing token.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

APPROVED_SUFFIXES = {
    ".safetensors", ".pt", ".pth", ".json", ".yaml", ".yml", ".txt",
    ".md", ".mp4", ".png", "_config.json", ".log",
}
CHUNK = 1 << 20


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def human(n: int) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}GB"


def redact(text: str, secrets: list[str]) -> str:
    out = str(text)
    for s in secrets or []:
        if s and s in out:
            out = out.replace(s, "<redacted>")
    return out[:300]


def detect_prefix_clash(existing_keys: list[str], prefix: str) -> list[str]:
    return [k for k in existing_keys if k.startswith(prefix.rstrip("/") + "/")]


def extract_remote_size(info) -> "int | None":
    """Remote size from RepoPathInfo: info.size primary, info.lfs.size fallback.

    None means 'unknown' (caller decides fail-closed). size=0 is valid.
    """
    size = getattr(info, "size", None)
    if size is None:
        lfs = getattr(info, "lfs", None)
        if lfs is not None:
            size = getattr(lfs, "size", None)
    if isinstance(size, (int, float)):
        return int(size)
    return None


def collect_files(step_dir: Path) -> list[Path]:
    """Fail-closed local file collection (symlinks, unapproved suffixes)."""
    if not step_dir.is_dir():
        raise FileNotFoundError(f"{step_dir}")
    for p in step_dir.iterdir():
        if p.is_symlink():
            raise ValueError(f"SYMLINK_REFUSED {p.name}")
    files = []
    for p in sorted(step_dir.iterdir()):
        if not p.is_file():
            continue
        if p.name.lower() == "manifest.json":
            raise ValueError("MANIFEST_IN_SOURCE_REFUSED manifest.json must not pre-exist in step dir")
        if p.suffix.lower() not in APPROVED_SUFFIXES:
            raise ValueError(f"UNAPPROVED_SUFFIX {p.name}")
        files.append(p)
    if not files:
        raise FileNotFoundError("NO_UPLOADABLE_FILES")
    return files


def build_manifest(run_id: str, step: int, tag: str, repo_id: str,
                   files: list[Path], step_dir: Path,
                   extra: dict | None = None) -> dict:
    return {
        "run_id": run_id,
        "step": step,
        "tag": tag,
        "repo_id": repo_id,
        "source_dir": str(step_dir),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "attempt_id": uuid.uuid4().hex[:8],
        "files": [
            {
                "name": p.name,
                "size": p.stat().st_size,
                "sha256": sha256_file(p),
            }
            for p in files
        ],
        "extra": extra or {},
    }


def verify_remote(api, repo_id: str, prefix: str, entries: list[dict],
                  manifest_expected_sha: str, secrets: list[str]) -> bool:
    """Verify remote paths+sizes (+LFS sha when provided) and manifest re-hash."""
    infos = api.get_paths_info(
        repo_id=repo_id, repo_type="model",
        paths=[f"{prefix}/{e['name']}" for e in entries],
    )
    info_by_path = {getattr(i, "path", None): i for i in infos}
    for e in entries:
        path = f"{prefix}/{e['name']}"
        info = info_by_path.get(path)
        if info is None:
            log(f"VERIFY_MISSING {e['name']}")
            return False
        got = extract_remote_size(info)
        if got is None:
            log(f"VERIFY_UNCOUNTED {e['name']}")
            return False
        if got != int(e["size"]):
            log(f"VERIFY_SIZE_MISMATCH {e['name']} want={e['size']} got={got}")
            return False
        lfs = getattr(info, "lfs", None)
        lfs_sha = getattr(lfs, "sha256", None) if lfs is not None else None
        if lfs_sha and lfs_sha.lower() != str(e["sha256"]).lower():
            log(f"VERIFY_LFS_SHA_MISMATCH {e['name']}")
            return False
        log(f"VERIFY_OK {e['name']} size={got}")
    # manifest.json re-hash via download (small file, direct read-back)
    with tempfile.TemporaryDirectory(prefix="hfcuver.") as td:
        local_manifest = Path(td) / "manifest.json"
        api.hf_hub_download(
            repo_id=repo_id, repo_type="model",
            filename=f"{prefix}/manifest.json",
            local_dir=td,
        )
        dpath = Path(td) / f"{prefix}" / "manifest.json"
        if not dpath.is_file():
            dpath = Path(td) / "manifest.json"
        back = sha256_file(dpath)
        if back.lower() != manifest_expected_sha.lower():
            log(f"VERIFY_MANIFEST_SHA_MISMATCH want={manifest_expected_sha[:12]} got={back[:12]}")
            return False
    return True


def audit_write(rec: dict) -> None:
    audit_dir = Path(os.environ.get("VISION_AUDIT_DIR", Path.home() / "vision-audit"))
    audit_dir.mkdir(parents=True, exist_ok=True)
    path = audit_dir / "upload-audit.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def die(code: int, msg: str) -> "int":
    log(f"FAILED {msg}")
    return code


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--step-dir", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--tag", default="adapter")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-manifest-check", action="store_true",
                    help="never used; reserved")
    args = ap.parse_args(argv)

    step_dir = Path(args.step_dir).resolve()
    prefix = f"checkpoints/step-{args.step:07d}-{args.run_id}"
    if args.tag:
        prefix += f"-{args.tag}"

    try:
        files = collect_files(step_dir)
    except Exception as e:
        return die(2, redact(e, []))

    entries = [
        {"name": p.name, "size": p.stat().st_size, "sha256": sha256_file(p)}
        for p in files
    ]
    total = sum(e["size"] for e in entries)
    log(f"PLAN step={args.step} tag={args.tag} files={len(entries)} bytes={total} ({human(total)})")
    for e in entries:
        log(f"  {e['name']} {e['size']} {e['sha256'][:12]}")
    log(f"PREFIX {prefix}")

    if args.dry_run:
        # manifest generated in a temp dir OUTSIDE the source dir to prove no mutation
        with tempfile.TemporaryDirectory(prefix="hfcudry.") as td:
            manifest = build_manifest(args.run_id, args.step, args.tag,
                                      args.repo_id, files, step_dir)
            (Path(td) / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True))
            log(f"DRY_RUN ok offline; manifest staged in temp dir {Path(td).name}")
        log("DRY_RUN_VERIFIED_LOCAL_ONLY")
        return 0

    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        audit_write({"ts": time.time(), "step": args.step, "run": args.run_id,
                     "status": "failed_no_token", "prefix": prefix})
        return die(4, "NO_TOKEN_IN_ENV")
    secrets = [token]

    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except Exception as e:
        return die(3, redact(("HUB_IMPORT_FAILED " + repr(e)), secrets))

    api = HfApi(token=token)
    try:
        info = api.repo_info(repo_id=args.repo_id, repo_type="model")
        if not getattr(info, "private", False):
            return die(3, "REPO_NOT_PRIVATE_REFUSED")
        sha_head = getattr(info, "sha", None)
    except Exception as e:
        audit_write({"ts": time.time(), "step": args.step, "run": args.run_id,
                     "status": "failed_repo_info", "prefix": prefix})
        return die(3, redact("REPO_INFO_FAILED " + repr(e), secrets))

    try:
        existing = api.list_repo_files(repo_id=args.repo_id, repo_type="model")
    except Exception as e:
        return die(3, redact("LIST_FAILED " + repr(e), secrets))
    clash = detect_prefix_clash(existing, prefix)
    if clash:
        return die(3, f"APPEND_ONLY_CLASH existing_files={len(clash)} first={clash[0]}")

    manifest = build_manifest(args.run_id, args.step, args.tag,
                              args.repo_id, files, step_dir)
    manifest_blob = json.dumps(manifest, indent=2, sort_keys=True).encode()
    manifest_sha = hashlib.sha256(manifest_blob).hexdigest()

    with tempfile.TemporaryDirectory(prefix="hfcumani.") as td:
        mpath = Path(td) / "manifest.json"
        mpath.write_bytes(manifest_blob)

        ops = [
            CommitOperationAdd(path_or_fileobj=str(p), path_in_repo=f"{prefix}/{p.name}")
            for p in files
        ]
        ops.append(
            CommitOperationAdd(path_or_fileobj=str(mpath), path_in_repo=f"{prefix}/manifest.json")
        )

        # Exactly ONE commit attempt. Uncertain commits are NEVER retried here.
        try:
            t0 = time.time()
            api.create_commit(
                repo_id=args.repo_id,
                repo_type="model",
                operations=ops,
                commit_message=f"ckp step-{args.step:07d} run-{args.run_id} ({args.tag})",
                parent_commit=sha_head,
            )
            log(f"COMMIT_OK seconds={time.time() - t0:.1f}")
        except Exception as e:
            audit_write({"ts": time.time(), "step": args.step, "run": args.run_id,
                         "status": "uncertain_commit", "prefix": prefix,
                         "err": redact(e, secrets)})
            return die(3, redact("UNCERTAIN_COMMIT no auto-retry; rerun gets a fresh prefix. " + repr(e), secrets))

    entries_with_manifest = entries + [{
        "name": "manifest.json", "size": len(manifest_blob), "sha256": manifest_sha,
    }]
    try:
        ok = verify_remote(api, args.repo_id, prefix, entries_with_manifest,
                           manifest_expected_sha=manifest_sha, secrets=secrets)
    except Exception as e:
        audit_write({"ts": time.time(), "step": args.step, "run": args.run_id,
                     "status": "verification_error", "prefix": prefix,
                     "err": redact(e, secrets)})
        return die(3, redact("VERIFY_ERROR " + repr(e), secrets))

    if not ok:
        audit_write({"ts": time.time(), "step": args.step, "run": args.run_id,
                     "status": "verification_failed", "prefix": prefix})
        return die(3, "VERIFY_NOT_VERIFIED — treat as FAILED; rerun uses fresh prefix")

    audit_write({"ts": time.time(), "step": args.step, "run": args.run_id,
                 "status": "verified", "prefix": prefix, "bytes": total,
                 "repo": args.repo_id})
    log("UPLOAD_VERIFIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
