import tempfile
import unittest
from pathlib import Path

from capstone_demo.store import RunStore, valid_run_id
from capstone_demo.uploads import validate_upload
from helpers import png_bytes


class StoreTests(unittest.TestCase):
    def test_run_id_validation_blocks_traversal(self):
        for bad in ("../etc", "a" * 31, "A" * 32, "", "0" * 32 + "/x", None):
            self.assertFalse(valid_run_id(bad), bad)
        self.assertTrue(valid_run_id("0123456789abcdef" * 2))

    def test_create_uses_server_side_names_and_lists_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(Path(tmp) / "runs")
            data = png_bytes()
            run = store.create(data, validate_upload(data, max_bytes=10**6))
            self.assertTrue(valid_run_id(run.run_id))
            self.assertEqual(run.input_path.name, "page.png")
            self.assertEqual(run.input_path.read_bytes(), data)
            self.assertEqual(run.input_path.parent.parent, store.root)
            store.write_result(run.run_id, {"run_id": run.run_id, "variants": []})
            self.assertEqual(store.read_result(run.run_id)["run_id"], run.run_id)
            self.assertEqual([r["run_id"] for r in store.list_results()], [run.run_id])
            self.assertIsNone(store.read_result("f" * 32))
            with self.assertRaises(ValueError):
                store.read_result("../../x")


if __name__ == "__main__":
    unittest.main()
