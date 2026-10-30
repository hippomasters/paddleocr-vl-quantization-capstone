"""Portable benchmark preflight contract tests."""

import json
import tempfile
import unittest
from pathlib import Path

from benchmark_driver import prepare_run, run_plan


def fixture(root):
    bundle = root / "bundle"
    bundle.mkdir()
    images = bundle / "images"
    images.mkdir()
    entries = []
    truth = []
    for index in range(1651):
        name = f"page_{index:04d}"
        (images / f"{name}.png").write_bytes(b"image")
        entries.append(
            {"annotation_index": index, "page_id": name, "image": f"images/{name}.png"}
        )
        truth.append({"page_info": {"image_path": f"images/{name}.png"}})
    (bundle / "benchmark_inputs.json").write_text(json.dumps(entries))
    (bundle / "OmniDocBench.json").write_text(json.dumps(truth))
    layout = bundle / "layout"
    layout.mkdir()
    for name in ("inference.json", "inference.pdiparams", "inference.yml"):
        (layout / name).write_text("layout")
    model = root / "my-model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"weights")
    (bundle / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "bundle": "paddleocr_vl_1_6_l4_wna16_bundle_v1",
                "paths": {
                    "benchmark_inputs": "benchmark_inputs.json",
                    "layout": "layout",
                    "omnidocbench": ".",
                    "bf16": "my-unused-model",
                },
            }
        )
    )
    return bundle, model


def cli(bundle, model, output, mode="PREFLIGHT", variant="BF16"):
    import subprocess
    import sys

    return subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts/benchmark_driver.py"),
            "--bundle",
            str(bundle),
            "--model",
            str(model),
            "--output",
            str(output),
            "--variant",
            variant,
            "--mode",
            mode,
            "--write-plan",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


class BenchmarkDriverTests(unittest.TestCase):
    def test_preflight_plan_uses_explicit_paths_and_nine_pages_in_c1_c24_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle, model = fixture(root)
            plan = prepare_run(bundle, model, root / "new-output", "BF16", "PREFLIGHT")
            self.assertEqual(
                plan["measured_indices"],
                [0, 103, 309, 584, 825, 1066, 1341, 1547, 1650],
            )
            self.assertEqual([run["concurrency"] for run in plan["runs"]], [1, 24])
            self.assertEqual(
                [run["output_dir"] for run in plan["runs"]],
                [str(root / "new-output/c1"), str(root / "new-output/c24")],
            )
            self.assertEqual(plan["checkpoint"], str(model))
            self.assertEqual(len(plan["expected_prediction_filenames"]), 9)
            self.assertFalse((root / "new-output").exists())

    def test_cli_writes_new_preflight_plan_only_after_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle, model = fixture(root)
            output = root / "new-output"
            result = cli(bundle, model, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads((output / "benchmark_plan.json").read_text())
            self.assertEqual(
                plan["measured_indices"],
                [0, 103, 309, 584, 825, 1066, 1341, 1547, 1650],
            )
            self.assertEqual(
                sorted(p.name for p in output.iterdir()), ["benchmark_plan.json"]
            )

    def test_full_mode_retains_all_pages_and_c1_c24_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle, model = fixture(root)
            plan = prepare_run(bundle, model, root / "out", "BF16", "FULL")
            self.assertEqual(plan["measured_indices"], list(range(1651)))
            self.assertEqual(len(plan["expected_prediction_filenames"]), 1651)
            self.assertEqual(plan["warmup_indices"], [34, 859, 1616])
            self.assertEqual(plan["concurrencies"], [1, 24])

    def test_rejects_missing_image_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle, model = fixture(root)
            (bundle / "images/page_0825.png").unlink()
            output = root / "out"
            result = cli(bundle, model, output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Benchmark image missing", result.stderr)
            self.assertFalse(output.exists())

    def test_rejects_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle, model = fixture(root)
            output = root / "out"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "Output path must be new"):
                prepare_run(bundle, model, output, "BF16", "PREFLIGHT")

    def test_rejects_ground_truth_name_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle, model = fixture(root)
            truth_path = bundle / "OmniDocBench.json"
            truth = json.loads(truth_path.read_text())
            truth[103]["page_info"]["image_path"] = "wrong.png"
            truth_path.write_text(json.dumps(truth))
            with self.assertRaisesRegex(ValueError, "Prediction-name mismatch"):
                prepare_run(bundle, model, root / "out", "BF16", "PREFLIGHT")

    def test_rejects_quantized_checkpoint_without_scope_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle, model = fixture(root)
            with self.assertRaisesRegex(
                ValueError, "Required file missing.*scope_audit.json"
            ):
                prepare_run(bundle, model, root / "out", "W8A16", "PREFLIGHT")

    def test_checkpoint_hashing_streams_without_read_bytes(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle, model = fixture(root)
            original = Path.read_bytes

            def reject_model_read(path):
                if path.parent == model:
                    raise AssertionError(
                        "Checkpoint read_bytes would load model into RAM"
                    )
                return original(path)

            with patch.object(Path, "read_bytes", reject_model_read):
                plan = prepare_run(bundle, model, root / "out", "BF16", "PREFLIGHT")
            self.assertEqual(len(plan["checkpoint_files"]), 2)

    def test_runner_starts_fresh_server_for_each_concurrency_and_stops_each(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle, model = fixture(root)
            plan = prepare_run(bundle, model, root / "out", "BF16", "PREFLIGHT")
            events = []

            def start(run):
                events.append(("start", run["concurrency"]))
                return object()

            def client(run, server):
                events.append(("client", run["concurrency"]))
                return {
                    "request_concurrency": run["concurrency"],
                    "page_count": len(plan["measured_indices"]),
                }

            def stop(server):
                events.append(("stop", server))

            results = run_plan(plan, start, client, stop)
            self.assertEqual([r["request_concurrency"] for r in results], [1, 24])
            self.assertEqual(
                [e[0] for e in events],
                ["start", "client", "stop", "start", "client", "stop"],
            )
            self.assertEqual([e[1] for e in events if e[0] == "start"], [1, 24])
            self.assertIsNot(events[2][1], events[5][1])

    def test_runner_stops_c1_and_does_not_start_c24_after_client_failure(self):
        events = []
        plan = {"runs": [{"concurrency": 1}, {"concurrency": 24}]}

        def start(run):
            events.append(("start", run["concurrency"]))
            return run["concurrency"]

        def client(run, server):
            events.append(("client", run["concurrency"]))
            raise RuntimeError("client failed")

        def stop(server):
            events.append(("stop", server))

        with self.assertRaisesRegex(RuntimeError, "client failed"):
            run_plan(plan, start, client, stop)
        self.assertEqual(events, [("start", 1), ("client", 1), ("stop", 1)])

    def test_quantized_variants_validate_existing_audit_and_bit_width(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle, model = fixture(root)
            audit = {
                "algorithm": "GPTQ",
                "group_size": 128,
                "activation_dtype": "BF16",
                "calibration_count": 512,
                "target_count": 234,
                "vision_layer_indices": list(range(27)),
                "decoder_layer_indices": list(range(18)),
            }
            (model / "scope_audit.json").write_text(json.dumps(audit))
            for variant, bits in (("W8A16", 8), ("W4A16", 4)):
                (model / "config.json").write_text(
                    json.dumps(
                        {
                            "quantization_config": {
                                "config_groups": [{"num_bits": bits, "group_size": 128}]
                            }
                        }
                    )
                )
                plan = prepare_run(
                    bundle, model, root / f"out-{variant}", variant, "PREFLIGHT"
                )
                self.assertEqual(plan["variant_key"], variant)
                self.assertEqual(len(plan["checkpoint_files"]), 3)
            audit["target_count"] = 233
            (model / "scope_audit.json").write_text(json.dumps(audit))
            with self.assertRaisesRegex(
                ValueError, "Checkpoint audit mismatch for target_count"
            ):
                prepare_run(bundle, model, root / "out-invalid", "W4A16", "PREFLIGHT")
