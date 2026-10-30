import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from verify_checkpoints import verify_checkpoint


class VerifyCheckpointTests(unittest.TestCase):
    def test_accepts_matching_model_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payloads = {"model.safetensors": b"model-bytes", "config.json": b"{}"}
            expected = {}
            for name, payload in payloads.items():
                (root / name).write_bytes(payload)
                expected[name] = {
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            result = verify_checkpoint(root, expected)
            self.assertEqual(result["verified_files"], sorted(payloads))

    def test_cli_accepts_explicit_manifest(self):
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            payload = b"checkpoint"
            (model / "model.safetensors").write_bytes(payload)
            manifest = root / "checksums.json"
            manifest.write_text(
                json.dumps(
                    {
                        "variants": {
                            "W8A16": {
                                "expected_files": {
                                    "model.safetensors": {
                                        "bytes": len(payload),
                                        "sha256": hashlib.sha256(payload).hexdigest(),
                                    }
                                }
                            }
                        }
                    }
                )
            )
            script = (
                Path(__file__).resolve().parents[1] / "scripts/verify_checkpoints.py"
            )
            run = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--manifest",
                    str(manifest),
                    "--variant",
                    "W8A16",
                    "--checkpoint",
                    str(model),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("model.safetensors", run.stdout)

    def test_rejects_unlisted_executable_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.safetensors").write_bytes(b"weights")
            (root / "unexpected.py").write_text("raise RuntimeError('unsafe')")
            expected = {
                "model.safetensors": {
                    "bytes": 7,
                    "sha256": hashlib.sha256(b"weights").hexdigest(),
                }
            }
            with self.assertRaisesRegex(ValueError, "Unlisted executable"):
                verify_checkpoint(root, expected)


if __name__ == "__main__":
    unittest.main()
