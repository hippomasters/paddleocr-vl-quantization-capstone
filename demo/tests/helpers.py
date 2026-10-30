"""Shared test fixtures. Fakes defined here are used ONLY by tests."""

import json
import sys
import struct
import zlib
from pathlib import Path


def _chunk(kind, data):
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))


def png_header(width, height):
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr)


def png_bytes(width=4, height=3):
    raw = b"".join(b"\x00" + b"\x00\x00\x00" * width for _ in range(height))
    return png_header(width, height) + _chunk(b"IDAT", zlib.compress(raw)) + _chunk(b"IEND", b"")


def jpeg_bytes(width=5, height=7):
    sof = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, height, width, 1) + b"\x01\x11\x00"
    return b"\xff\xd8" + b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00" + sof + b"\xff\xd9"


def pdf_bytes():
    return b"%PDF-1.7\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"


def make_host_tree(root: Path) -> dict:
    """Create placeholder host paths so config readiness checks pass."""
    root = Path(root)
    # Real interpreter path only satisfies readiness; tests never execute it.
    pythons = {"vllm_python": sys.executable, "paddle_python": sys.executable}
    layout = root / "layout"
    layout.mkdir()
    for name in ("inference.json", "inference.pdiparams", "inference.yml"):
        (layout / name).write_bytes(b"x")
    models = {}
    for variant in ("BF16", "W8A16", "W4A16"):
        model = root / variant
        model.mkdir()
        (model / "config.json").write_text("{}")
        (model / "model.safetensors").write_bytes(b"w")
        models[variant] = str(model)
    return {**pythons, "layout_dir": str(layout), "models": models}


def write_config(root: Path, **overrides) -> Path:
    data = make_host_tree(root)
    data.update(overrides)
    path = Path(root) / "demo_config.json"
    path.write_text(json.dumps(data))
    return path


class FakeRunner:
    """Test double for the process layer. Never imported by production code."""

    def __init__(self, fail_ready=(), fail_client=(), skip_outputs=()):
        self.events = []
        self.fail_ready = set(fail_ready)
        self.fail_client = set(fail_client)
        self.skip_outputs = set(skip_outputs)
        self.active = None

    def port_in_use(self, port):
        return False

    def start_server(self, command, env, log_path):
        variant = command[command.index("--variant") + 1]
        assert self.active is None, "server started while another is running"
        self.active = variant
        Path(log_path).write_text(f"fake server log {variant}\n")
        self.events.append(("start", variant))
        return variant

    def wait_ready(self, handle, port, served_name, timeout_s):
        self.events.append(("ready", handle))
        if handle in self.fail_ready:
            raise RuntimeError("server exited before becoming ready")

    def run_client(self, command, env, log_path, timeout_s):
        variant = command[command.index("--variant-label") + 1]
        self.events.append(("client", variant))
        Path(log_path).write_text("fake client log\n")
        if variant in self.fail_client:
            return 3
        if variant in self.skip_outputs:
            return 0
        out = Path(command[command.index("--output-dir") + 1])
        out.mkdir(parents=True)
        stem = Path(command[command.index("--image") + 1]).stem
        (out / f"{stem}.md").write_text(f"# TEST-ONLY {variant}\n")
        (out / f"{stem}.json").write_text(json.dumps({"parsing_res_list": [1, 2]}))
        timing = Path(command[command.index("--timing-out") + 1])
        timing.write_text(
            json.dumps(
                {
                    "pipeline_init_s": 1.5,
                    "passes": [
                        {"kind": "warmup", "seconds": 2.0},
                        {"kind": "measured", "seconds": 0.75},
                    ],
                    "measured_parse_s": 0.75,
                }
            )
        )
        return 0

    def stop_server(self, handle):
        self.events.append(("stop", handle))
        self.active = None

    def terminate_all(self):
        self.events.append(("terminate",))

