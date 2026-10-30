import json
import tempfile
import unittest
from pathlib import Path

from capstone_demo.config import load_config
from capstone_demo.orchestrator import (
    Orchestrator,
    build_client_command,
    build_server_command,
    client_environment,
    parse_nvidia_smi,
    server_environment,
)
from capstone_demo.paths import CLIENT_SCRIPT, MANIFEST, SERVE_SCRIPT
from capstone_demo.store import RunStore
from capstone_demo.uploads import validate_upload
from helpers import FakeRunner, png_bytes, write_config


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = load_config(write_config(self.root))

    def tearDown(self):
        self.tmp.cleanup()

    def test_server_command_delegates_to_verified_launcher(self):
        command = build_server_command(self.config, "W4A16")
        self.assertIsInstance(command, list)
        self.assertTrue(all(isinstance(part, str) for part in command))
        self.assertEqual(command[:2], [self.config.vllm_python, str(SERVE_SCRIPT)])
        pairs = dict(zip(command[2::2], command[3::2]))
        self.assertEqual(pairs["--variant"], "W4A16")
        self.assertEqual(pairs["--served-model-name"], "GPTQ_W4A16_G128")
        self.assertEqual(pairs["--manifest"], str(MANIFEST))
        self.assertEqual(pairs["--model"], self.config.models["W4A16"])
        self.assertEqual(pairs["--gpu-memory-utilization"], str(self.config.gpu_memory_utilization))
        self.assertNotIn("--verify-only", command)
        self.assertEqual(
            build_server_command(self.config, "BF16")[
                build_server_command(self.config, "BF16").index("--served-model-name") + 1
            ],
            "BF16_W16A16",
        )

    def test_client_command_uses_tested_parser_and_local_server(self):
        image = self.root / "runs" / "r" / "page.png"
        command = build_client_command(self.config, "W8A16", image, self.root / "out")
        self.assertEqual(command[:2], [self.config.paddle_python, str(CLIENT_SCRIPT)])
        pairs = dict(zip(command[2::2], command[3::2]))
        self.assertEqual(pairs["--server-url"], f"http://127.0.0.1:{self.config.vllm_port}/v1")
        self.assertEqual(pairs["--device"], "cpu")
        self.assertEqual(pairs["--served-model-name"], "GPTQ_W8A16_G128")
        self.assertEqual(pairs["--image"], str(image))
        self.assertEqual(pairs["--warmup-passes"], "1")

    def test_environments_are_offline_and_cpu_layout_hides_gpu(self):
        server = server_environment({"HTTP_PROXY": "http://proxy"})
        client = client_environment(self.config, {"HTTPS_PROXY": "http://proxy"})
        for env in (server, client):
            self.assertEqual(env["HF_HUB_OFFLINE"], "1")
            self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
            self.assertIn("127.0.0.1", env["NO_PROXY"])
            self.assertNotIn("HTTP_PROXY", env)
            self.assertNotIn("HTTPS_PROXY", env)
        self.assertEqual(server["VLLM_NO_USAGE_STATS"], "1")
        self.assertEqual(client["CUDA_VISIBLE_DEVICES"], "")

    def test_parse_nvidia_smi(self):
        self.assertEqual(parse_nvidia_smi("6123, 8188\n"), (6123, 8188))
        self.assertIsNone(parse_nvidia_smi("garbage"))


class OrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = load_config(write_config(self.root))
        self.store = RunStore(self.root / "runs")
        data = png_bytes()
        self.run = self.store.create(data, validate_upload(data, max_bytes=10**6))

    def tearDown(self):
        self.tmp.cleanup()

    def orchestrate(self, runner):
        return Orchestrator(self.config, self.store, runner=runner, gpu_sampler=None).run(self.run)

    def test_runs_three_variants_strictly_sequentially(self):
        runner = FakeRunner()
        result = self.orchestrate(runner)
        expected = []
        for variant in ("BF16", "W8A16", "W4A16"):
            expected += [("start", variant), ("ready", variant), ("client", variant), ("stop", variant)]
        self.assertEqual(runner.events, expected)
        self.assertEqual(result["status"], "done")
        statuses = [v["status"] for v in result["variants"]]
        self.assertEqual(statuses, ["done", "done", "done"])
        w8 = result["variants"][1]
        self.assertEqual(w8["markdown"], "# TEST-ONLY W8A16\n")
        self.assertEqual(w8["timings"]["measured_parse_s"], 0.75)
        self.assertEqual(w8["timings"]["first_parse_s"], 2.0)
        self.assertEqual(w8["json_regions"], 2)
        self.assertIsNone(result["variants"][0]["similarity_to_bf16"])
        self.assertIsInstance(w8["similarity_to_bf16"], float)
        self.assertEqual(self.store.read_result(self.run.run_id), result)

    def test_failed_server_is_stopped_and_next_variant_still_runs(self):
        runner = FakeRunner(fail_ready={"W8A16"})
        result = self.orchestrate(runner)
        self.assertIn(("stop", "W8A16"), runner.events)
        self.assertNotIn(("client", "W8A16"), runner.events)
        w8 = result["variants"][1]
        self.assertEqual(w8["status"], "failed")
        self.assertIn("ready", w8["error"])
        self.assertIn("fake server log", w8["error_log_tail"])
        self.assertIsNone(w8["markdown"])
        self.assertEqual(result["variants"][2]["status"], "done")
        self.assertEqual(result["status"], "partial")

    def test_missing_outputs_are_reported_never_fabricated(self):
        result = self.orchestrate(FakeRunner(skip_outputs={"W4A16"}, fail_client={"BF16"}))
        bf16, _, w4 = result["variants"]
        self.assertEqual(bf16["status"], "failed")
        self.assertIn("exit code 3", bf16["error"])
        self.assertEqual(w4["status"], "failed")
        self.assertIsNone(w4["markdown"])
        self.assertIsNone(result["variants"][1]["similarity_to_bf16"])

    def test_refuses_to_launch_when_port_already_serving(self):
        runner = FakeRunner()
        runner.port_in_use = lambda port: True
        result = self.orchestrate(runner)
        self.assertEqual(runner.events, [])
        self.assertTrue(all(v["status"] == "failed" for v in result["variants"]))
        self.assertIn("already in use", result["variants"][0]["error"])

    def test_abort_stops_active_server(self):
        runner = FakeRunner()
        orchestrator = Orchestrator(self.config, self.store, runner=runner, gpu_sampler=None)
        original = runner.run_client

        def client_then_abort(*args, **kwargs):
            orchestrator.abort()
            return original(*args, **kwargs)

        runner.run_client = client_then_abort
        result = orchestrator.run(self.run)
        self.assertEqual(runner.events[-1], ("stop", "BF16"))
        self.assertNotIn(("start", "W8A16"), runner.events)
        self.assertEqual(result["status"], "aborted")
        self.assertEqual(result["variants"][1]["status"], "skipped")

    def test_result_records_provenance_not_claims(self):
        result = self.orchestrate(FakeRunner())
        self.assertEqual(result["input"]["kind"], "png")
        self.assertEqual(result["settings"]["layout_device"], "cpu")
        self.assertEqual(result["settings"]["layout_vlm_same_gpu_validated"], False)
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
