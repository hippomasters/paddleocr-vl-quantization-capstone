import http.client
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from capstone_demo.app import make_server
from capstone_demo.config import load_config
from capstone_demo.orchestrator import Orchestrator
from capstone_demo.paths import STATIC_DIR
from capstone_demo.store import RunStore
from helpers import FakeRunner, png_bytes, write_config


class AppTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.config = load_config(write_config(root, max_upload_mb=1))
        self.store = RunStore(root / "runs")
        self.gate = threading.Event()
        runner = FakeRunner()
        original = runner.run_client

        def gated_client(*args, **kwargs):
            self.gate.wait(5)
            return original(*args, **kwargs)

        runner.run_client = gated_client
        self.orchestrator = Orchestrator(self.config, self.store, runner=runner, gpu_sampler=None)
        self.server = make_server(self.config, self.store, self.orchestrator, port=0)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.gate.set()
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def request(self, method, path, body=None, headers=None, host=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        hdrs = {"Host": host or f"127.0.0.1:{self.port}"}
        if method == "POST":
            hdrs["X-Capstone-Demo"] = "1"
            hdrs["X-Capstone-Variants"] = "BF16,W8A16,W4A16"
        hdrs.update(headers or {})
        conn.request(method, path, body=body, headers=hdrs)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response, data

    def test_binds_loopback_only(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")

    def test_index_has_strict_security_headers(self):
        response, body = self.request("GET", "/")
        self.assertEqual(response.status, 200)
        csp = response.getheader("Content-Security-Policy")
        self.assertIn("default-src 'none'", csp)
        self.assertIn("script-src 'self'", csp)
        self.assertNotIn("unsafe-inline", csp)
        self.assertEqual(response.getheader("X-Content-Type-Options"), "nosniff")
        self.assertIn(b"<script src=\"/app.js\"", body)

    def test_rejects_foreign_host_header_and_cross_origin_posts(self):
        response, _ = self.request("GET", "/", host="evil.example:80")
        self.assertEqual(response.status, 421)
        response, _ = self.request("POST", "/api/runs", body=png_bytes(), headers={"X-Capstone-Demo": ""})
        self.assertEqual(response.status, 403)
        response, _ = self.request(
            "POST", "/api/runs", body=png_bytes(), headers={"Origin": "http://evil.example"}
        )
        self.assertEqual(response.status, 403)

    def test_upload_limits_and_type_checks(self):
        response, _ = self.request(
            "POST", "/api/runs", body=b"x", headers={"Content-Length": str(2 * 1024 * 1024)}
        )
        self.assertEqual(response.status, 413)
        response, body = self.request("POST", "/api/runs", body=b"<svg onload=alert(1)>")
        self.assertEqual(response.status, 415)
        self.assertEqual(response.getheader("Content-Type"), "application/json")

    def test_traversal_and_unknown_paths_404(self):
        for path in ("/api/runs/../../etc/passwd", "/api/runs/%2e%2e%2fconfig", "/static/../app.py", "/demo_config.json"):
            response, _ = self.request("GET", path)
            self.assertEqual(response.status, 404, path)

    def test_full_run_lifecycle_and_single_gpu_job_lock(self):
        response, body = self.request("POST", "/api/runs", body=png_bytes())
        self.assertEqual(response.status, 202, body)
        run_id = json.loads(body)["run_id"]
        response, _ = self.request("POST", "/api/runs", body=png_bytes())
        self.assertEqual(response.status, 409)
        self.gate.set()
        deadline = time.time() + 5
        while time.time() < deadline:
            response, body = self.request("GET", f"/api/runs/{run_id}")
            result = json.loads(body)
            if result["status"] in ("done", "partial", "failed"):
                break
            time.sleep(0.05)
        self.assertEqual(result["status"], "done")
        response, body = self.request("GET", f"/api/runs/{run_id}/input")
        self.assertEqual(response.getheader("Content-Type"), "image/png")
        self.assertEqual(body, png_bytes())
        response, body = self.request("GET", "/api/runs")
        self.assertEqual([r["run_id"] for r in json.loads(body)["runs"]], [run_id])

    def test_selected_single_variant_is_only_model_launched(self):
        response, body = self.request("POST", "/api/runs", body=png_bytes(), headers={"X-Capstone-Variants": "W4A16"})
        self.assertEqual(response.status, 202, body)
        run_id = json.loads(body)["run_id"]
        self.gate.set()
        self.server.job_thread.join(5)
        response, body = self.request("GET", f"/api/runs/{run_id}")
        result = json.loads(body)
        self.assertEqual(result["status"], "done")
        self.assertEqual([v["status"] for v in result["variants"]], ["skipped", "skipped", "done"])
        self.assertIsNone(result["variants"][2]["similarity_to_bf16"])
        self.assertEqual(self.orchestrator.runner.events, [
            ("start", "W4A16"), ("ready", "W4A16"), ("client", "W4A16"), ("stop", "W4A16")
        ])

    def test_empty_unknown_duplicate_or_missing_selection_is_rejected_before_run_creation(self):
        for selection in ("", "BF16,", "W4A16,W4A16", "INVALID", "W8A16,INVALID"):
            response, _ = self.request("POST", "/api/runs", body=png_bytes(), headers={"X-Capstone-Variants": selection})
            self.assertEqual(response.status, 400, selection)
        self.assertEqual(self.store.list_results(), [])

    def test_downloads_only_completed_run_variant_outputs_as_attachments(self):
        response, body = self.request("POST", "/api/runs", body=png_bytes(), headers={"X-Capstone-Variants": "W8A16"})
        self.assertEqual(response.status, 202)
        run_id = json.loads(body)["run_id"]
        self.gate.set()
        self.server.job_thread.join(5)
        for kind, content_type, expected in (
            ("markdown", "text/plain; charset=utf-8", b"# TEST-ONLY W8A16\n"),
            ("json", "application/json", b'{"parsing_res_list": [1, 2]}'),
        ):
            response, body = self.request("GET", f"/api/runs/{run_id}/outputs/W8A16/{kind}")
            self.assertEqual(response.status, 200, body)
            self.assertEqual(response.getheader("Content-Type"), content_type)
            self.assertIn("attachment", response.getheader("Content-Disposition"))
            self.assertEqual(response.getheader("X-Content-Type-Options"), "nosniff")
            self.assertEqual(body, expected)
        for path in (
            f"/api/runs/{run_id}/outputs/BF16/markdown",
            f"/api/runs/{run_id}/outputs/W8A16/logs",
            f"/api/runs/{run_id}/outputs/W8A16/../../page.png",
            f"/api/runs/{run_id}/outputs/W8A16/%2e%2e",
            f"/api/runs/{'0' * 32}/outputs/W8A16/json",
        ):
            response, _ = self.request("GET", path)
            self.assertEqual(response.status, 404, path)
        output = self.store.root / run_id / "W8A16" / "page.md"
        output.unlink()
        secret = Path(self.tmp.name) / "secret.txt"
        secret.write_text("secret")
        output.symlink_to(secret)
        response, _ = self.request("GET", f"/api/runs/{run_id}/outputs/W8A16/markdown")
        self.assertEqual(response.status, 404)

    def test_status_reports_readiness_and_limited_laptop_gpu_validation(self):
        response, body = self.request("GET", "/api/status")
        status = json.loads(body)
        self.assertEqual(status["problems"], [])
        self.assertFalse(status["busy"])
        self.assertEqual(status["variant_problems"], {"BF16": [], "W8A16": [], "W4A16": []})
        self.assertIn("synthetic page GPU-tested", status["adapter_note"])
        self.assertIn("arbitrary pages and PDFs remain unverified", status["adapter_note"])

    def test_not_ready_host_refuses_runs_instead_of_faking(self):
        Path(self.config.models["BF16"], "model.safetensors").unlink()
        response, body = self.request("POST", "/api/runs", body=png_bytes())
        self.assertEqual(response.status, 503)
        self.assertIn("BF16", json.loads(body)["error"])

    def test_unselected_unready_model_does_not_block_selected_ready_model(self):
        Path(self.config.models["BF16"], "model.safetensors").unlink()
        response, body = self.request("POST", "/api/runs", body=png_bytes(), headers={"X-Capstone-Variants": "W4A16"})
        self.assertEqual(response.status, 202, body)
        self.gate.set()
        self.server.job_thread.join(5)
        self.assertEqual(self.store.read_result(json.loads(body)["run_id"])["status"], "done")


class StaticSafetyTests(unittest.TestCase):
    def test_ui_exposes_explicit_model_selection_and_text_outputs_with_downloads(self):
        html = (STATIC_DIR / "index.html").read_text()
        script = (STATIC_DIR / "app.js").read_text()
        for variant in ("BF16", "W8A16", "W4A16"):
            self.assertIn(f'value="{variant}"', html)
        self.assertIn('type="checkbox"', html)
        self.assertIn('name="variant"', html)
        self.assertIn('X-Capstone-Variants', script)
        self.assertIn('input[name="variant"]:checked', script)
        self.assertIn('JSON.stringify(v.structured_json', script)
        self.assertIn('el("pre"', script)
        self.assertIn('/outputs/${v.variant}/${kind}', script)
        self.assertIn('[["markdown", "Download Markdown"], ["json", "Download JSON"]]', script)
        self.assertIn('updateRunButton(busy);', script)
        self.assertNotIn('updateRunButton(false);', script)
        self.assertNotIn("innerHTML", script)

    def test_frontend_never_renders_untrusted_html(self):
        script = (STATIC_DIR / "app.js").read_text()
        for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
            self.assertNotIn(sink, script, sink)
        html = (STATIC_DIR / "index.html").read_text()
        self.assertNotIn("<script>", html)
        self.assertNotIn("http://", html.replace("http://www.w3.org", ""))
        self.assertNotIn("https://", html)


if __name__ == "__main__":
    unittest.main()
