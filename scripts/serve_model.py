"""Serve a verified PaddleOCR-VL-1.6 checkpoint with the study's vLLM settings."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def build_server_command(
    model: Path,
    name: str,
    port: int,
    *,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    quantized: bool,
    python_executable: str = sys.executable,
) -> list[str]:
    command = [
        python_executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model),
        "--served-model-name",
        name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--generation-config",
        "vllm",
        "--no-enable-log-requests",
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb",
        "0",
        "--seed",
        "2026",
    ]
    if quantized:
        command += ["--linear-backend", "marlin"]
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--variant", required=True, choices=["BF16", "W8A16", "W4A16"])
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check all recorded model files without launching vLLM",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config/checkpoint_hashes.json",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=24)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    args = parser.parse_args()
    config_path = args.model / "config.json"
    if not config_path.is_file() or not (args.model / "model.safetensors").is_file():
        parser.error("model directory must contain config.json and model.safetensors")
    if not 0 < args.gpu_memory_utilization <= 1 or args.port < 1 or args.port > 65535:
        parser.error("invalid port or GPU-memory-utilization")
    if (
        min(args.max_model_len, args.max_num_seqs, args.max_num_batched_tokens) < 1
        or args.gpu_index < 0
    ):
        parser.error(
            "model length, sequence count, batched tokens, and GPU index "
            "must be positive"
        )
    from verify_checkpoints import verify_checkpoint

    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        expected = manifest["variants"][args.variant]["expected_files"]
        verified = verify_checkpoint(args.model, expected)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        parser.error(f"Verification failed: {exc}")
    if args.verify_only:
        print(f"Verified {len(verified['verified_files'])} files for {args.variant}")
        return 0
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quantized = isinstance(config.get("quantization_config"), dict)
    command = build_server_command(
        args.model.resolve(),
        args.served_model_name,
        args.port,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        quantized=quantized,
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    environment["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    return subprocess.run(command, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
