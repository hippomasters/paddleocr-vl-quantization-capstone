import unittest
from pathlib import Path
from evaluate_official import build_docker_command


class OfficialEvaluatorTests(unittest.TestCase):
    def test_docker_command_mounts_inputs_read_only(self):
        cmd = build_docker_command(
            Path("/gt.json"),
            Path("/predictions"),
            Path("/results"),
            Path("/config.yaml"),
            "omnidoc-test",
        )
        self.assertEqual(cmd[:3], ["docker", "run", "--rm"])
        self.assertIn("--network", cmd)
        self.assertEqual(cmd[cmd.index("--network") + 1], "none")
        self.assertTrue(
            any(x.endswith(":/workspace/gt/OmniDocBench.json:ro") for x in cmd)
        )
        self.assertTrue(
            any(x.endswith(":/workspace/data_md/predictions:ro") for x in cmd)
        )
        self.assertTrue(any("omnidocbench-eval@sha256:" in x for x in cmd))
        self.assertIn("pdf_validation.py --config /tmp/eval.yaml", cmd[-1])

    def test_inputs_keep_the_empty_page(self):
        import json
        import tempfile
        from evaluate_official import validate_inputs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred = root / "pred"
            pred.mkdir()
            gt = root / "ground_truth.json"
            gt.write_text(
                json.dumps(
                    [
                        {"page_info": {"image_path": "images/first.png"}},
                        {"page_info": {"image_path": "images/second.jpg"}},
                    ]
                )
            )
            (pred / "first.md").write_text("recognized")
            (pred / "second.md").write_text("")
            result = validate_inputs(gt, pred, expected_pages=2)
            self.assertEqual(result["prediction_files"], 2)
            self.assertEqual(result["empty_filenames"], ["second.md"])

    def test_cli_dry_run_validates_without_docker(self):
        import json
        import subprocess
        import sys
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred = root / "pred"
            pred.mkdir()
            gt = root / "ground_truth.json"
            gt.write_text(
                json.dumps(
                    [
                        {"page_info": {"image_path": "first.png"}},
                        {"page_info": {"image_path": "second.png"}},
                    ]
                )
            )
            (pred / "first.md").write_text("one")
            (pred / "second.md").write_text("")
            output = root / "output"
            script = (
                Path(__file__).resolve().parents[1] / "scripts/evaluate_official.py"
            )
            run = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--ground-truth",
                    str(gt),
                    "--predictions",
                    str(pred),
                    "--output-dir",
                    str(output),
                    "--expected-pages",
                    "2",
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout)["prediction_files"], 2)
            self.assertFalse(output.exists())

    def test_rejects_output_inside_predictions(self):
        import tempfile
        from evaluate_official import validate_output_location

        with tempfile.TemporaryDirectory() as tmp:
            predictions = Path(tmp) / "predictions"
            predictions.mkdir()
            with self.assertRaisesRegex(ValueError, "inside predictions"):
                validate_output_location(predictions, predictions / "results")
            self.assertFalse((predictions / "results").exists())


if __name__ == "__main__":
    unittest.main()
