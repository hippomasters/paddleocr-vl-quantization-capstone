import json
import tempfile
import types
import unittest
from pathlib import Path

from capstone_demo import timed_parse
from helpers import png_bytes


class FakeResult:
    def __init__(self, text):
        self.text = text

    def save_to_markdown(self, path, pretty, show_formula_number):
        Path(path).write_text(self.text)

    def save_to_json(self, path):
        Path(path).write_text(json.dumps({"parsing_res_list": []}))


class FakePipeline:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append((image, kwargs))
        return [FakeResult("TEST-ONLY")]


class FakeNoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_paddle():
    module = types.SimpleNamespace(seeds=[], synced=[])
    module.seed = module.seeds.append
    module.no_grad = FakeNoGrad
    module.device = types.SimpleNamespace(synchronize=lambda: module.synced.append(1))
    return module


class TimedParseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.image = root / "page.png"
        self.image.write_bytes(png_bytes())
        self.layout = root / "layout"
        self.layout.mkdir()
        for name in ("inference.json", "inference.pdiparams", "inference.yml"):
            (self.layout / name).write_bytes(b"x")
        self.out = root / "W8A16"
        self.timing = root / "timing.json"
        self.argv = [
            "--image", str(self.image), "--layout-dir", str(self.layout),
            "--server-url", "http://127.0.0.1:8000/v1", "--served-model-name",
            "GPTQ_W8A16_G128", "--output-dir", str(self.out), "--timing-out",
            str(self.timing), "--device", "cpu", "--variant-label", "W8A16",
            "--warmup-passes", "1",
        ]

    def tearDown(self):
        self.tmp.cleanup()

    def test_uses_parse_page_contract_and_records_warm_latency(self):
        paddle = fake_paddle()
        pipelines = []

        def factory(**kwargs):
            pipelines.append(FakePipeline(**kwargs))
            return pipelines[-1]

        ticks = iter([0.0, 10.0, 10.0, 12.5, 12.5, 13.25])
        code = timed_parse.main(self.argv, paddle_module=paddle, pipeline_factory=factory, clock=lambda: next(ticks))
        self.assertEqual(code, 0)
        self.assertEqual(paddle.seeds, [2026])
        self.assertEqual(paddle.synced, [])
        from parse_page import PREDICT_KWARGS, build_pipeline_kwargs

        self.assertEqual(
            pipelines[0].kwargs,
            build_pipeline_kwargs(self.layout, "http://127.0.0.1:8000/v1", "GPTQ_W8A16_G128", 1, "cpu"),
        )
        self.assertEqual(len(pipelines[0].calls), 2)
        self.assertEqual(pipelines[0].calls[1][1], PREDICT_KWARGS)
        timing = json.loads(self.timing.read_text())
        self.assertEqual(timing["pipeline_init_s"], 10.0)
        self.assertEqual([p["kind"] for p in timing["passes"]], ["warmup", "measured"])
        self.assertEqual(timing["measured_parse_s"], 0.75)
        self.assertTrue((self.out / "page.md").is_file())
        self.assertFalse((self.out / "page.md").read_text() == "")

    def test_refuses_existing_outputs_and_non_local_server(self):
        self.timing.write_text("{}")
        with self.assertRaises(SystemExit):
            timed_parse.main(self.argv, paddle_module=fake_paddle(), pipeline_factory=FakePipeline)
        self.timing.unlink()
        argv = list(self.argv)
        argv[argv.index("--server-url") + 1] = "http://example.org/v1"
        with self.assertRaises(SystemExit):
            timed_parse.main(argv, paddle_module=fake_paddle(), pipeline_factory=FakePipeline)


if __name__ == "__main__":
    unittest.main()
