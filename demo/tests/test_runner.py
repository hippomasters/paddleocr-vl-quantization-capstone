"""Exercise the real subprocess layer against a test-only stub HTTP process."""

import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path

from capstone_demo.app import main as app_main
from capstone_demo.orchestrator import SubprocessRunner
from helpers import write_config

STUB = r"""
import http.server, json, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"data": [{"id": sys.argv[2]}]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
"""


DELAYED_CHILD = r"""
import http.server, json, signal, sys, time
def finish(signum, frame):
    time.sleep(0.8)
    raise SystemExit(0)
signal.signal(signal.SIGTERM, finish)
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"data": [{"id": "NAME"}]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
"""
PARENT = "import subprocess,sys; subprocess.run([sys.executable,'-c',sys.argv[2],sys.argv[1]])"

def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class SubprocessRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = Path(self.tmp.name) / "server.log"
        self.runner = SubprocessRunner()

    def tearDown(self):
        self.runner.terminate_all()
        self.tmp.cleanup()

    def test_ready_poll_then_process_group_stop_frees_port(self):
        port = free_port()
        self.assertFalse(self.runner.port_in_use(port))
        handle = self.runner.start_server([sys.executable, "-c", STUB, str(port), "NAME"], {}, self.log)
        self.runner.wait_ready(handle, port, "NAME", timeout_s=20)
        self.assertTrue(self.runner.port_in_use(port))
        self.runner.stop_server(handle)
        self.assertIsNotNone(handle.poll())
        time.sleep(0.2)
        self.assertFalse(self.runner.port_in_use(port))

    def test_stop_waits_for_grandchild_to_release_port(self):
        port = free_port()
        handle = self.runner.start_server(
            [sys.executable, "-c", PARENT, str(port), DELAYED_CHILD], {}, self.log
        )
        try:
            self.runner.wait_ready(handle, port, "NAME", timeout_s=20)
            self.runner.stop_server(handle, port=port)
            self.assertFalse(self.runner.port_in_use(port))
        finally:
            if handle.poll() is None:
                self.runner.stop_server(handle)

    def test_wrong_served_name_never_counts_as_ready(self):
        port = free_port()
        handle = self.runner.start_server([sys.executable, "-c", STUB, str(port), "OTHER"], {}, self.log)
        try:
            with self.assertRaisesRegex(RuntimeError, "not ready"):
                self.runner.wait_ready(handle, port, "NAME", timeout_s=3)
        finally:
            self.runner.stop_server(handle)

    def test_early_exit_is_reported_with_code(self):
        handle = self.runner.start_server([sys.executable, "-c", "raise SystemExit(2)"], {}, self.log)
        with self.assertRaisesRegex(RuntimeError, "exited with code 2"):
            self.runner.wait_ready(handle, free_port(), "NAME", timeout_s=20)

    def test_client_exit_code_and_arguments_are_not_shell_interpreted(self):
        marker = Path(self.tmp.name) / "pwned"
        code = self.runner.run_client(
            [sys.executable, "-c", "import sys; sys.exit(len(sys.argv))", f"; touch {marker}"],
            {}, self.log, timeout_s=20,
        )
        self.assertEqual(code, 2)
        self.assertFalse(marker.exists())


class AppCliTests(unittest.TestCase):
    def test_ui_port_must_differ_from_vllm_port(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = write_config(Path(tmp))
            with self.assertRaises(SystemExit):
                app_main(["--config", str(config), "--port", "8000", "--data-dir", tmp])


if __name__ == "__main__":
    unittest.main()


class AppSignalTests(unittest.TestCase):
    def test_sigterm_shuts_down_cleanly(self):
        import signal
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            config = write_config(Path(tmp))
            port = free_port()
            process = subprocess.Popen(
                [sys.executable, "-m", "capstone_demo.app", "--config", str(config),
                 "--port", str(port), "--data-dir", tmp],
                cwd=Path(__file__).resolve().parents[1],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            import threading

            guard = threading.Timer(20, process.kill)
            guard.start()
            try:
                self.assertIn(b"Serving", process.stdout.readline())
                process.send_signal(signal.SIGTERM)
                self.assertEqual(process.wait(15), 0)
            finally:
                guard.cancel()
                if process.poll() is None:
                    process.kill()
                process.stdout.close()
