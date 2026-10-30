import json
import tempfile
import unittest
from pathlib import Path

from capstone_demo.config import ConfigError, load_config, readiness
from helpers import make_host_tree, write_config


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, data):
        path = self.root / "c.json"
        path.write_text(json.dumps(data))
        return path

    def test_defaults_target_one_8gb_gpu_with_cpu_layout(self):
        config = load_config(write_config(self.root))
        self.assertEqual(config.layout_device, "cpu")
        self.assertEqual(config.variant_order, ("BF16", "W8A16", "W4A16"))
        self.assertLess(config.gpu_memory_utilization, 0.82)
        self.assertEqual(config.max_upload_bytes, 20 * 1024 * 1024)

    def test_rejects_relative_paths_unknown_keys_and_missing_variants(self):
        base = make_host_tree(self.root)
        bad = [
            {**base, "layout_dir": "relative/layout"},
            {**base, "surprise": 1},
            {**base, "models": {"BF16": base["models"]["BF16"]}},
            {**base, "models": {**base["models"], "W2A16": "/x"}},
        ]
        for data in bad:
            with self.assertRaises(ConfigError, msg=data):
                load_config(self.write(data))

    def test_rejects_unsafe_device_and_out_of_range_numbers(self):
        base = make_host_tree(self.root)
        bad = [
            {**base, "layout_device": "cpu; rm -rf /"},
            {**base, "layout_device": "gpu"},
            {**base, "vllm_port": 70000},
            {**base, "gpu_memory_utilization": 1.5},
            {**base, "max_num_seqs": 0},
            {**base, "max_upload_mb": 500},
            {**base, "warmup": "yes"},
        ]
        for data in bad:
            with self.assertRaises(ConfigError, msg=data):
                load_config(self.write(data))

    def test_gpu_layout_device_is_allowed_explicitly(self):
        config = load_config(self.write({**make_host_tree(self.root), "layout_device": "gpu:0"}))
        self.assertEqual(config.layout_device, "gpu:0")

    def test_readiness_reports_missing_host_paths(self):
        config = load_config(write_config(self.root))
        self.assertEqual(readiness(config), [])
        (Path(config.models["W4A16"]) / "model.safetensors").unlink()
        (Path(config.layout_dir) / "inference.yml").unlink()
        problems = readiness(config)
        self.assertTrue(any("W4A16" in p for p in problems), problems)
        self.assertTrue(any("inference.yml" in p for p in problems), problems)

    def test_readiness_checks_only_selected_model_but_shared_layout(self):
        config = load_config(write_config(self.root))
        (Path(config.models["BF16"]) / "model.safetensors").unlink()
        self.assertEqual(readiness(config, ("W4A16",)), [])
        self.assertTrue(any("BF16" in p for p in readiness(config, ("BF16",))))
        (Path(config.layout_dir) / "inference.yml").unlink()
        self.assertTrue(any("inference.yml" in p for p in readiness(config, ("W4A16",))))


if __name__ == "__main__":
    unittest.main()
