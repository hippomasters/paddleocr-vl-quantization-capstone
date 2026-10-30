"""Sequential BF16 -> W8A16 -> W4A16 execution on one GPU.

Each variant: launch source/scripts/serve_model.py (which re-hashes every
manifest file, including custom model code, before --trust-remote-code), wait
until the local endpoint lists the served name, run the timed parse client,
then stop the server so the next variant gets the whole GPU.
"""

from datetime import datetime, timezone
import difflib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

from .paths import CLIENT_SCRIPT, MANIFEST, SERVE_SCRIPT, SOURCE_SCRIPTS

if str(SOURCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SOURCE_SCRIPTS))
from benchmark_driver import VARIANTS  # noqa: E402

MARKDOWN_DISPLAY_LIMIT = 256 * 1024
SIMILARITY_LIMIT = 20000
PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "NO_PROXY": "127.0.0.1,localhost",
    "no_proxy": "127.0.0.1,localhost",
}


def served_name(variant):
    return VARIANTS[variant][0]


def _offline(base):
    env = {k: v for k, v in dict(base).items() if k not in PROXY_KEYS}
    env.update(OFFLINE_ENV)
    return env


def server_environment(base=None):
    env = _offline(os.environ if base is None else base)
    env["VLLM_NO_USAGE_STATS"] = "1"
    return env


def client_environment(config, base=None):
    env = _offline(os.environ if base is None else base)
    env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    if config.layout_device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(config.gpu_index)
    return env


def build_server_command(config, variant):
    return [
        config.vllm_python,
        str(SERVE_SCRIPT),
        "--model", config.models[variant],
        "--served-model-name", served_name(variant),
        "--variant", variant,
        "--manifest", str(MANIFEST),
        "--port", str(config.vllm_port),
        "--gpu-index", str(config.gpu_index),
        "--gpu-memory-utilization", str(config.gpu_memory_utilization),
        "--max-model-len", str(config.max_model_len),
        "--max-num-seqs", str(config.max_num_seqs),
        "--max-num-batched-tokens", str(config.max_num_batched_tokens),
    ]


def build_client_command(config, variant, image, run_dir):
    run_dir = Path(run_dir)
    return [
        config.paddle_python,
        str(CLIENT_SCRIPT),
        "--image", str(image),
        "--layout-dir", config.layout_dir,
        "--server-url", f"http://127.0.0.1:{config.vllm_port}/v1",
        "--served-model-name", served_name(variant),
        "--output-dir", str(run_dir / variant),
        "--timing-out", str(run_dir / f"{variant}.timing.json"),
        "--device", config.layout_device,
        "--variant-label", variant,
        "--warmup-passes", "1" if config.warmup else "0",
    ]


def parse_nvidia_smi(text):
    try:
        used, total = (int(part.strip()) for part in text.strip().splitlines()[0].split(","))
    except (ValueError, IndexError):
        return None
    return used, total


class NvidiaSmiSampler:
    """Polls device-wide memory.used (includes display and other processes)."""

    def __init__(self, gpu_index, interval_s=0.5):
        self.command = [
            shutil.which("nvidia-smi") or "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
            "-i", str(gpu_index),
        ]
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = None
        self.peak = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(self.command, capture_output=True, text=True, timeout=5, check=False)
                sample = parse_nvidia_smi(out.stdout)
            except (OSError, subprocess.SubprocessError):
                sample = None
            if sample and (self.peak is None or sample[0] > self.peak[0]):
                self.peak = sample
            self._stop.wait(self.interval_s)

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(10)
        if self.peak is None:
            return None
        return {"peak_used_mib": self.peak[0], "total_mib": self.peak[1]}


def default_gpu_sampler(config):
    if shutil.which("nvidia-smi") is None:
        return None
    return lambda: NvidiaSmiSampler(config.gpu_index)


class SubprocessRunner:
    """Real process layer: argument lists only, no shell, own process groups."""

    def __init__(self):
        self._lock = threading.Lock()
        self._processes = set()

    def port_in_use(self, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    def _spawn(self, command, env, log_path):
        log = open(log_path, "ab")
        try:
            process = subprocess.Popen(
                command, env=env, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
            )
        finally:
            log.close()
        with self._lock:
            self._processes.add(process)
        return process

    def _kill(self, process, grace_s):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(grace_s)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(30)
            except ProcessLookupError:
                pass
        with self._lock:
            self._processes.discard(process)

    def start_server(self, command, env, log_path):
        return self._spawn(command, env, log_path)

    def wait_ready(self, handle, port, served_name, timeout_s):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            code = handle.poll()
            if code is not None:
                raise RuntimeError(f"server exited with code {code} before becoming ready")
            try:
                with opener.open(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
                    models = json.loads(response.read(1024 * 1024))
                if served_name in {m.get("id") for m in models.get("data", [])}:
                    return
            except (OSError, ValueError, AttributeError):
                pass
            time.sleep(1)
        raise RuntimeError(f"server not ready within {timeout_s}s")

    def run_client(self, command, env, log_path, timeout_s):
        process = self._spawn(command, env, log_path)
        try:
            return process.wait(timeout_s)
        except subprocess.TimeoutExpired:
            self._kill(process, 10)
            raise RuntimeError(f"parser timed out after {timeout_s}s")
        finally:
            with self._lock:
                self._processes.discard(process)

    def stop_server(self, handle):
        self._kill(handle, 60)

    def terminate_all(self):
        with self._lock:
            processes = list(self._processes)
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass


def _tail(path, limit=4000):
    try:
        data = Path(path).read_bytes()[-limit:]
    except OSError:
        return ""
    return data.decode("utf-8", "replace")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Aborted(Exception):
    pass


class Orchestrator:
    def __init__(self, config, store, *, runner, gpu_sampler):
        self.config = config
        self.store = store
        self.runner = runner
        self.gpu_sampler = gpu_sampler
        self._abort = threading.Event()

    def abort(self):
        self._abort.set()
        self.runner.terminate_all()

    def _initial(self, run):
        info = run.info
        c = self.config
        return {
            "run_id": run.run_id,
            "created_utc": _now(),
            "status": "running",
            "input": {
                "kind": info.kind, "bytes": info.size, "sha256": info.sha256,
                "width": info.width, "height": info.height,
            },
            "settings": {
                "layout_device": c.layout_device,
                "layout_vlm_same_gpu_validated": False,
                "gpu_memory_utilization": c.gpu_memory_utilization,
                "max_model_len": c.max_model_len,
                "max_num_seqs": c.max_num_seqs,
                "max_num_batched_tokens": c.max_num_batched_tokens,
                "warmup": c.warmup,
                "concurrency": 1,
            },
            "variants": [
                {
                    "variant": v, "served_name": served_name(v), "status": "pending",
                    "error": None, "error_log_tail": None, "timings": {},
                    "gpu_memory": None, "markdown": None, "markdown_truncated": False,
                    "json_regions": None, "similarity_to_bf16": None,
                }
                for v in c.variant_order
            ],
        }

    def _collect(self, entry, run, variant):
        out = run.directory / variant
        stem = run.input_path.stem
        markdown, structured = out / f"{stem}.md", out / f"{stem}.json"
        timing_path = run.directory / f"{variant}.timing.json"
        for path in (markdown, structured, timing_path):
            if not path.is_file():
                raise RuntimeError(f"parser exited 0 but did not write {path.name}")
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        passes = timing.get("passes") or []
        entry["timings"].update(
            pipeline_init_s=timing.get("pipeline_init_s"),
            first_parse_s=passes[0]["seconds"] if passes else None,
            measured_parse_s=timing.get("measured_parse_s"),
        )
        text = markdown.read_text(encoding="utf-8", errors="replace")
        entry["markdown"] = text[:MARKDOWN_DISPLAY_LIMIT]
        entry["markdown_truncated"] = len(text) > MARKDOWN_DISPLAY_LIMIT
        regions = json.loads(structured.read_text(encoding="utf-8")).get("parsing_res_list")
        entry["json_regions"] = len(regions) if isinstance(regions, list) else None
        entry["outputs"] = {"markdown": str(markdown.relative_to(run.directory)), "json": str(structured.relative_to(run.directory))}

    def _run_variant(self, run, result, entry):
        variant = entry["variant"]
        c = self.config
        logs = run.directory / "logs"
        logs.mkdir(exist_ok=True)
        server_log, client_log = logs / f"{variant}.server.log", logs / f"{variant}.client.log"
        entry["logs"] = {"server": f"logs/{server_log.name}", "client": f"logs/{client_log.name}"}
        entry["status"] = "starting_server"
        self.store.write_result(run.run_id, result)
        if self.runner.port_in_use(c.vllm_port):
            raise RuntimeError(
                f"port {c.vllm_port} already in use; refusing to measure an unknown server"
            )
        sampler = self.gpu_sampler() if self.gpu_sampler else None
        if sampler:
            sampler.start()
        handle = None
        try:
            began = time.perf_counter()
            handle = self.runner.start_server(build_server_command(c, variant), server_environment(), server_log)
            self.runner.wait_ready(handle, c.vllm_port, served_name(variant), c.server_ready_timeout_s)
            entry["timings"]["server_ready_s"] = time.perf_counter() - began
            if self._abort.is_set():
                raise Aborted()
            entry["status"] = "parsing"
            self.store.write_result(run.run_id, result)
            began = time.perf_counter()
            code = self.runner.run_client(
                build_client_command(c, variant, run.input_path, run.directory),
                client_environment(c), client_log, c.client_timeout_s,
            )
            entry["timings"]["client_wall_s"] = time.perf_counter() - began
            if code != 0:
                raise RuntimeError(f"parser failed with exit code {code}")
            self._collect(entry, run, variant)
            entry["status"] = "done"
        finally:
            if handle is not None:
                self.runner.stop_server(handle)
            if sampler:
                entry["gpu_memory"] = sampler.stop()

    def run(self, run):
        result = self._initial(run)
        self.store.write_result(run.run_id, result)
        for entry in result["variants"]:
            if self._abort.is_set():
                entry["status"] = "skipped"
                continue
            try:
                self._run_variant(run, result, entry)
            except Aborted:
                entry["status"] = "skipped"
            except Exception as exc:  # report every failure; never substitute output
                entry.update(status="failed", error=str(exc), markdown=None)
                logs = entry.get("logs", {})
                entry["error_log_tail"] = "\n".join(
                    _tail(run.directory / logs[k]) for k in ("server", "client") if k in logs
                ) or None
            self.store.write_result(run.run_id, result)
        reference = result["variants"][0]["markdown"]
        for entry in result["variants"][1:]:
            if reference is not None and entry["markdown"] is not None:
                matcher = difflib.SequenceMatcher(None, reference[:SIMILARITY_LIMIT], entry["markdown"][:SIMILARITY_LIMIT])
                entry["similarity_to_bf16"] = round(matcher.ratio(), 4)
        statuses = {e["status"] for e in result["variants"]}
        if self._abort.is_set():
            result["status"] = "aborted"
        elif statuses == {"done"}:
            result["status"] = "done"
        elif "done" in statuses:
            result["status"] = "partial"
        else:
            result["status"] = "failed"
        result["finished_utc"] = _now()
        self.store.write_result(run.run_id, result)
        return result
