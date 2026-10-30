import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from summarize_performance import load_run


class PerformanceTests(unittest.TestCase):
    def test_recomputes_run_from_individual_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.zip"
            rows = [
                {
                    "page_id": "first",
                    "page_inference_seconds": 2.0,
                    "vlm_seconds": 1.0,
                    "generated_tokens": 10,
                    "token_limit_reached": False,
                    "status": "completed",
                },
                {
                    "page_id": "second",
                    "page_inference_seconds": 4.0,
                    "vlm_seconds": 2.0,
                    "generated_tokens": 20,
                    "token_limit_reached": True,
                    "status": "completed",
                },
            ]
            with zipfile.ZipFile(path, "w") as z:
                z.writestr(
                    "c1/page_metrics.jsonl", "".join(json.dumps(x) + "\n" for x in rows)
                )
            run = load_run(path, "c1", expected_pages=2)
            self.assertEqual(run["page_count"], 2)
            self.assertEqual(run["total_page_seconds"], 6.0)
            self.assertEqual(run["total_tokens"], 30)
            self.assertEqual(run["token_limit_pages"], 1)

    def test_summary_and_paired_speedup(self):
        from summarize_performance import summarize_run, paired_speedup

        def run(times):
            pages = {
                name: {"page_inference_seconds": t}
                for name, t in zip(["first", "second"], times, strict=True)
            }
            return {
                "page_count": 2,
                "pages": pages,
                "total_page_seconds": sum(times),
                "total_vlm_seconds": 3,
                "total_tokens": 30,
                "token_limit_pages": 0,
                "completed_pages": 2,
            }

        baseline, quantized = run([4, 8]), run([2, 4])
        metrics = summarize_run(quantized)
        self.assertEqual(metrics["pages_per_hour"], 3600 * 2 / 6)
        self.assertAlmostEqual(metrics["p95_page_seconds"], 3.9)
        self.assertEqual(paired_speedup(baseline, quantized), 2.0)

    def test_cli_exposes_three_variant_inputs(self):
        import subprocess
        import sys

        script = (
            Path(__file__).resolve().parents[1] / "scripts/summarize_performance.py"
        )
        run = subprocess.run(
            [sys.executable, str(script), "--help"], capture_output=True, text=True
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        for flag in ["--bf16", "--w8a16", "--w4a16"]:
            self.assertIn(flag, run.stdout)

    def test_rejects_different_c1_and_c24_cohorts(self):
        from summarize_performance import validate_common_cohort

        runs = {
            variant: {
                "c1": {"pages": {"first": {}}},
                "c24": {"pages": {"different": {}}},
            }
            for variant in ["BF16", "W8A16", "W4A16"]
        }
        with self.assertRaisesRegex(ValueError, "same page cohort"):
            validate_common_cohort(runs)

    def test_rejects_nan_page_time(self):
        import tempfile
        import zipfile

        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "bad.zip"
            row = {
                "page_id": "first",
                "page_inference_seconds": float("nan"),
                "vlm_seconds": 1,
                "generated_tokens": 1,
                "token_limit_reached": False,
                "status": "completed",
            }
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("c1/page_metrics.jsonl", json.dumps(row))
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                load_run(archive, "c1", expected_pages=1)


if __name__ == "__main__":
    unittest.main()
