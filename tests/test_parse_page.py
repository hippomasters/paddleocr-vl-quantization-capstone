import unittest
from pathlib import Path
from parse_page import build_pipeline_kwargs, PREDICT_KWARGS


class ParsePageTests(unittest.TestCase):
    def test_pipeline_matches_measured_layout_and_vlm_options(self):
        kwargs = build_pipeline_kwargs(
            Path("/models/layout"),
            "https://example.org/v1",
            "GPTQ_W8A16_G128",
            24,
            "gpu:0",
        )
        self.assertEqual(kwargs["layout_detection_model_name"], "PP-DocLayoutV3")
        self.assertEqual(kwargs["vl_rec_backend"], "vllm-server")
        self.assertEqual(kwargs["vl_rec_max_concurrency"], 24)
        self.assertTrue(kwargs["merge_layout_blocks"])
        self.assertFalse(kwargs["use_doc_unwarping"])
        self.assertEqual(PREDICT_KWARGS["max_new_tokens"], 4096)
        self.assertEqual(PREDICT_KWARGS["max_pixels"], 1003520)

    def test_parse_one_image_saves_both_outputs(self):
        import tempfile
        import json
        from parse_page import parse_one_image

        class Result:
            def save_to_markdown(self, path, pretty, show_formula_number):
                Path(path).write_text("# parsed", encoding="utf-8")

            def save_to_json(self, path):
                Path(path).write_text('{"ok": true}', encoding="utf-8")

        class Pipeline:
            def predict(self, image, **kwargs):
                self.image, self.kwargs = image, kwargs
                return iter([Result()])

        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "page.png"
            image.write_bytes(b"image")
            output = Path(tmp) / "output"
            pipeline = Pipeline()
            result = parse_one_image(pipeline, image, output)
            self.assertEqual(Path(result["markdown"]).read_text(), "# parsed")
            self.assertEqual(json.loads(Path(result["json"]).read_text()), {"ok": True})
            self.assertEqual(pipeline.kwargs["max_new_tokens"], 4096)
            self.assertEqual(pipeline.image, str(image))

    def test_cli_help_does_not_require_paddle(self):
        import subprocess
        import sys

        script = Path(__file__).resolve().parents[1] / "scripts/parse_page.py"
        run = subprocess.run(
            [sys.executable, str(script), "--help"], capture_output=True, text=True
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("--layout-dir", run.stdout)
        self.assertIn("--server-url", run.stdout)


if __name__ == "__main__":
    unittest.main()
