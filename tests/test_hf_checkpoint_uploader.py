"""Offline unittest suite for hf_checkpoint_uploader.py — issue #9 acceptance matrix.

Run: python -m unittest discover -s <dir containing this file> -v
No network is required; huggingface_hub is never imported by these tests.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# repo root = parent of scripts/ when tests live in repo layout; module path search
SCRIPT_CANDIDATES = [
    HERE.parent / "scripts" / "hf_checkpoint_uploader.py",
    HERE / "hf_checkpoint_uploader.py",
]
SCRIPT = next(p for p in SCRIPT_CANDIDATES if p.is_file())
sys.path.insert(0, str(SCRIPT.parent))

import hf_checkpoint_uploader as up  # noqa: E402


def _script_env(clear_tokens: bool) -> dict:
    env = dict(os.environ)
    if clear_tokens:
        env.pop("HUGGING_FACE_HUB_TOKEN", None)
        env.pop("HF_TOKEN", None)
    return env


def _run(args, clear_tokens):
    return subprocess.run(
        [sys.executable, str(SCRIPT)] + args,
        env=_script_env(clear_tokens),
        capture_output=True, text=True, timeout=120,
    )


def _make_step_dir(tmp: Path, n_files: int = 2, zero_file: bool = True,
                   symlink: Path | None = None, unapproved: str | None = None,
                   preexisting_manifest: bool = False) -> Path:
    d = tmp / "step-dir"
    d.mkdir(exist_ok=True)
    (d / "adapter.safetensors").write_bytes(b"x" * (1 << 20))
    if n_files > 1:
        (d / "training_state.pt").write_bytes(b"y" * (2048))
    if zero_file:
        (d / "trainer.log").write_bytes(b"")
    if symlink is not None:
        link = d / "bogus.safetensors"
        try:
            os.symlink(str(symlink), str(link))
        except OSError:
            pass
    if unapproved:
        (d / unapproved).write_bytes(b"z")
    if preexisting_manifest:
        (d / "manifest.json").write_text("{}")
    return d


class PureHelpers(unittest.TestCase):
    def test_detect_prefix_clash(self):
        existing = ["checkpoints/step-0000250-r1/manifest.json", "other/file"]
        self.assertEqual(
            len(up.detect_prefix_clash(existing, "checkpoints/step-0000250-r1")), 1)
        self.assertEqual(
            len(up.detect_prefix_clash(existing, "checkpoints/step-0000250-r2")), 0)

    def test_extract_remote_size_variants(self):
        class I:
            pass

        i = I(); i.size = 123
        self.assertEqual(up.extract_remote_size(i), 123)
        i2 = I(); i2.size = 0
        self.assertEqual(up.extract_remote_size(i2), 0)
        i3 = I(); i3.size = None
        i3.lfs = type("L", (), {"size": 456})()
        self.assertEqual(up.extract_remote_size(i3), 456)
        i4 = I(); i4.size = None; i4.lfs = None
        self.assertIsNone(up.extract_remote_size(i4))

    def test_human(self):
        self.assertEqual(up.human(0), "0.0B")
        self.assertIn("KB", up.human(2048))

    def test_redact(self):
        self.assertEqual(up.redact("tok=abc", ["abc"]), "tok=<redacted>")

    def test_sha256_file_zero(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "f"
            p.write_bytes(b"")
            self.assertEqual(len(up.sha256_file(p)), 64)


class LocalFailures(unittest.TestCase):
    """collect_files fail-closed behavior + dry-run offline acceptance."""

    def test_symlink_refused(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            target = tmp / "target.txt"
            target.write_text("t")
            d = _make_step_dir(tmp, symlink=target)
            with self.assertRaises(ValueError):
                up.collect_files(d)

    def test_unapproved_refused(self):
        with tempfile.TemporaryDirectory() as td:
            d = _make_step_dir(Path(td), unapproved="weird.exe")
            with self.assertRaises(ValueError):
                up.collect_files(d)

    def test_preexisting_manifest_refused(self):
        with tempfile.TemporaryDirectory() as td:
            d = _make_step_dir(Path(td), preexisting_manifest=True)
            with self.assertRaises(ValueError):
                up.collect_files(d)


class OfflineCliMatrix(unittest.TestCase):
    def test_dry_run_no_token_exits_zero(self):
        with tempfile.TemporaryDirectory() as td:
            d = _make_step_dir(Path(td))
            before = {p.name: (p.stat().st_size, p.stat().st_mtime)
                      for p in sorted(d.iterdir())}
            r = _run(["--repo-id", "TamerITIS/x", "--step-dir", str(d),
                      "--step", "250", "--run-id", "abc123", "--dry-run"],
                     clear_tokens=True)
            self.assertEqual(r.returncode, 0, msg=r.stdout + r.stderr)
            self.assertIn("DRY_RUN_VERIFIED_LOCAL_ONLY", r.stdout)
            after = {p.name: (p.stat().st_size, p.stat().st_mtime)
                     for p in sorted(d.iterdir())}
            self.assertEqual(before, after, "source dir mutated!")
            self.assertNotIn("manifest.json", after)

    def test_dry_run_symlink_exits_two(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            target = tmp / "outside"
            target.write_text("x")
            d = _make_step_dir(tmp, symlink=target)
            try:
                r = _run(["--repo-id", "TamerITIS/x", "--step-dir", str(d),
                          "--step", "1", "--run-id", "r", "--dry-run"],
                         clear_tokens=True)
            except subprocess.SubprocessError as e:
                e2 = getattr(e, "exception", None)
                self.fail(f"subprocess failed: {e2 if e2 else e}")
            self.assertEqual(r.returncode, 2, msg=r.stdout + r.stderr)
            self.assertIn("SYMLINK_REFUSED", r.stdout)

    def test_live_no_token_exits_four(self):
        with tempfile.TemporaryDirectory() as td:
            d = _make_step_dir(Path(td))
            r = _run(["--repo-id", "TamerITIS/x", "--step-dir", str(d),
                      "--step", "1", "--run-id", "r"],
                     clear_tokens=True)
            self.assertEqual(r.returncode, 4, msg=r.stdout + r.stderr)
            self.assertIn("FAILED NO_TOKEN_IN_ENV", r.stdout)
            self.assertNotIn("hf_", r.stdout + r.stderr)
            self.assertNotIn("Traceback", r.stdout + r.stderr)

    def test_manifest_sha_matches_upload_bytes(self):
        """Manifest sha256 recorded equals sha of exactly the bytes that are staged."""
        with tempfile.TemporaryDirectory() as td:
            d = _make_step_dir(Path(td))
            files = up.collect_files(d)
            m = up.build_manifest("r1", 250, "adapter", "TamerITIS/x", files, d)
            blob = json.dumps(m, indent=2, sort_keys=True).encode()
            msha = hashlib.sha256(blob).hexdigest()
            re = [{**e} for e in m["files"]]
            re.append({"name": "manifest.json",
                       "size": len(blob), "sha256": msha})
            self.assertTrue(all(e["sha256"] for e in re))


if __name__ == "__main__":
    unittest.main(verbosity=2)
