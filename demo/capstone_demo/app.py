"""Localhost-only web UI for the sequential three-checkpoint page demo.

Usage: python -m capstone_demo.app --config demo_config.json [--port 8765]
"""

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import signal
import sys
import threading
from urllib.parse import urlsplit

from .config import ConfigError, load_config, readiness
from .orchestrator import Orchestrator, SubprocessRunner, default_gpu_sampler
from .paths import DEMO_DIR, STATIC_DIR
from .store import RunStore
from .uploads import UploadError, validate_upload

HOST = "127.0.0.1"
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}
RUN_PATH = re.compile(r"/api/runs/([0-9a-f]{32})(/input)?")
OUTPUT_PATH = re.compile(r"/api/runs/([0-9a-f]{32})/outputs/(BF16|W8A16|W4A16)/(markdown|json)")
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' blob:; "
        "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Cache-Control": "no-store",
}
ADAPTER_NOTE = (
    "Real adapter: source/scripts/serve_model.py (hash-verifies all model files and "
    "custom code before trust_remote_code) + parse_page.py pipeline. The demo package "
    "is not GPU-tested; single-GPU layout+VLM is not validated. Results shown are only "
    "what this host actually produced."
)


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port, handler, config, store, orchestrator):
        super().__init__((HOST, port), handler)
        self.config = config
        self.store = store
        self.orchestrator = orchestrator
        self.job_lock = threading.Lock()
        self.job_thread = None

    def allowed_hosts(self):
        port = self.server_address[1]
        return {f"127.0.0.1:{port}", f"localhost:{port}"}


class Handler(BaseHTTPRequestHandler):
    server_version = "CapstoneDemo"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    def _send(self, status, body: bytes, content_type, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in {**SECURITY_HEADERS, **(extra or {})}.items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status, payload):
        self._send(status, json.dumps(payload).encode(), "application/json")

    def _error(self, status, message):
        self._json(status, {"error": message})

    def _host_ok(self):
        if self.headers.get("Host") not in self.server.allowed_hosts():
            self.close_connection = True
            self._error(HTTPStatus.MISDIRECTED_REQUEST, "Unexpected Host header")
            return False
        return True

    def do_GET(self):
        if not self._host_ok():
            return
        path = urlsplit(self.path).path
        if path in STATIC:
            name, content_type = STATIC[path]
            return self._send(HTTPStatus.OK, (STATIC_DIR / name).read_bytes(), content_type)
        store = self.server.store
        if path == "/api/status":
            config = self.server.config
            return self._json(
                HTTPStatus.OK,
                {
                    "problems": readiness(config),
                    "busy": self.server.job_lock.locked(),
                    "adapter_note": ADAPTER_NOTE,
                    "variants": list(config.variant_order),
                    "variant_problems": {v: readiness(config, (v,)) for v in config.variant_order},
                    "layout_device": config.layout_device,
                    "max_upload_bytes": config.max_upload_bytes,
                },
            )
        if path == "/api/runs":
            return self._json(HTTPStatus.OK, {"runs": store.list_results()})
        output_match = OUTPUT_PATH.fullmatch(path)
        if output_match:
            run_id, variant, kind = output_match.groups()
            source = store.output_file(run_id, variant, kind)
            if source is None:
                return self._error(HTTPStatus.NOT_FOUND, "Not found")
            content_type = "text/plain; charset=utf-8" if kind == "markdown" else "application/json"
            filename = f"{variant}-page.{ 'md' if kind == 'markdown' else 'json' }"
            return self._send(HTTPStatus.OK, source.read_bytes(), content_type,
                              {"Content-Disposition": f'attachment; filename="{filename}"'})
        match = RUN_PATH.fullmatch(path)
        if not match or not store.exists(match.group(1)):
            return self._error(HTTPStatus.NOT_FOUND, "Not found")
        run_id = match.group(1)
        if match.group(2):
            source = store.input_file(run_id)
            if source is None:
                return self._error(HTTPStatus.NOT_FOUND, "Not found")
            types = {".png": "image/png", ".jpg": "image/jpeg", ".pdf": "application/pdf"}
            extra = {"Content-Disposition": "attachment"} if source.suffix == ".pdf" else {}
            return self._send(HTTPStatus.OK, source.read_bytes(), types[source.suffix], extra)
        result = store.read_result(run_id) or {"run_id": run_id, "status": "queued", "variants": []}
        return self._json(HTTPStatus.OK, result)

    def do_POST(self):
        if not self._host_ok():
            return
        self.close_connection = True
        origin = self.headers.get("Origin")
        allowed = {f"http://{host}" for host in self.server.allowed_hosts()}
        if self.headers.get("X-Capstone-Demo") != "1" or (origin and origin not in allowed):
            return self._error(HTTPStatus.FORBIDDEN, "Cross-origin or unmarked request refused")
        if urlsplit(self.path).path != "/api/runs":
            return self._error(HTTPStatus.NOT_FOUND, "Not found")
        selected_header = self.headers.get("X-Capstone-Variants")
        selected = selected_header.split(",") if selected_header is not None else []
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(self.server.config.variant_order):
            return self._error(HTTPStatus.BAD_REQUEST, "Select one or more distinct known models")
        if self.headers.get("Transfer-Encoding"):
            return self._error(HTTPStatus.LENGTH_REQUIRED, "Chunked uploads not accepted")
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            return self._error(HTTPStatus.LENGTH_REQUIRED, "Content-Length required")
        config = self.server.config
        if length > config.max_upload_bytes:
            return self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"Upload exceeds {config.max_upload_mb} MB")
        if length < 1:
            return self._error(HTTPStatus.BAD_REQUEST, "Empty upload")
        data = self.rfile.read(length)
        try:
            info = validate_upload(data, max_bytes=config.max_upload_bytes)
        except UploadError as exc:
            return self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, str(exc))
        problems = readiness(config, selected)
        if problems:
            return self._error(HTTPStatus.SERVICE_UNAVAILABLE, "Host not ready: " + "; ".join(problems))
        if not self.server.job_lock.acquire(blocking=False):
            return self._error(HTTPStatus.CONFLICT, "A run is already using the GPU")
        try:
            run = self.server.store.create(data, info)
        except Exception:
            self.server.job_lock.release()
            raise

        def job():
            try:
                self.server.orchestrator.run(run, selected)
            finally:
                self.server.job_lock.release()

        self.server.job_thread = threading.Thread(target=job, name=f"run-{run.run_id}")
        self.server.job_thread.start()
        self._json(HTTPStatus.ACCEPTED, {"run_id": run.run_id})

    def log_message(self, format, *args):
        sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), format % args))


def make_server(config, store, orchestrator, *, port):
    return DemoServer(port, Handler, config, store, orchestrator)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path, default=DEMO_DIR / "runs")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        parser.error(str(exc))
    if not 1024 <= args.port <= 65535 or args.port == config.vllm_port:
        parser.error("UI port must be in [1024, 65535] and differ from vllm_port")
    store = RunStore(args.data_dir)
    orchestrator = Orchestrator(
        config, store, runner=SubprocessRunner(), gpu_sampler=default_gpu_sampler(config)
    )
    server = make_server(config, store, orchestrator, port=args.port)
    for problem in readiness(config):
        print(f"NOT READY: {problem}", file=sys.stderr)
    print(f"Serving http://127.0.0.1:{args.port}/ (loopback only). Ctrl+C stops any running model server.", flush=True)

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    # Closing the terminal or `kill` must also stop a vLLM child holding the GPU.
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupt)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        orchestrator.abort()
        server.server_close()
        if server.job_thread is not None:
            server.job_thread.join(120)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
