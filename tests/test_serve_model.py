import unittest
from pathlib import Path
from serve_model import build_server_command


class ServeModelTests(unittest.TestCase):
    def test_quantized_command_matches_measured_runtime_flags(self):
        command = build_server_command(
            Path("/models/W8A16"),
            "GPTQ_W8A16_G128",
            8011,
            gpu_memory_utilization=0.82,
            max_model_len=8192,
            max_num_seqs=24,
            max_num_batched_tokens=16384,
            quantized=True,
            python_executable="python",
        )
        self.assertEqual(
            command[:4],
            ["python", "-m", "vllm.entrypoints.openai.api_server", "--model"],
        )
        self.assertIn("--trust-remote-code", command)
        self.assertEqual(command[command.index("--linear-backend") + 1], "marlin")
        self.assertEqual(
            command[command.index("--served-model-name") + 1], "GPTQ_W8A16_G128"
        )
        self.assertIn("--no-enable-prefix-caching", command)
        self.assertEqual(command[command.index("--mm-processor-cache-gb") + 1], "0")

    def test_cli_refuses_unverified_model_code(self):
        import hashlib
        import json
        import subprocess
        import sys
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            config = b'{"quantization_config":{}}'
            weights = b"weights"
            (model / "config.json").write_bytes(config)
            (model / "model.safetensors").write_bytes(weights)
            expected = {
                name: {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
                for name, value in {
                    "config.json": config,
                    "model.safetensors": weights,
                    "modeling_paddleocr_vl.py": b"missing",
                }.items()
            }
            manifest = root / "checksums.json"
            manifest.write_text(
                json.dumps({"variants": {"W8A16": {"expected_files": expected}}})
            )
            script = Path(__file__).resolve().parents[1] / "scripts/serve_model.py"
            run = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--model",
                    str(model),
                    "--served-model-name",
                    "test",
                    "--variant",
                    "W8A16",
                    "--manifest",
                    str(manifest),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("Verification failed", run.stderr)

    def test_cli_verify_only_does_not_launch_vllm(self):
        import hashlib
        import json
        import subprocess
        import sys
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            payloads = {
                "config.json": b'{"quantization_config":{}}',
                "model.safetensors": b"weights",
                "modeling_paddleocr_vl.py": b"pass",
            }
            expected = {}
            for name, payload in payloads.items():
                (model / name).write_bytes(payload)
                expected[name] = {
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            manifest = root / "checksums.json"
            manifest.write_text(
                json.dumps({"variants": {"W8A16": {"expected_files": expected}}})
            )
            script = Path(__file__).resolve().parents[1] / "scripts/serve_model.py"
            run = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--model",
                    str(model),
                    "--served-model-name",
                    "test",
                    "--variant",
                    "W8A16",
                    "--manifest",
                    str(manifest),
                    "--verify-only",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("Verified 3 files", run.stdout)


if __name__ == "__main__":
    unittest.main()
