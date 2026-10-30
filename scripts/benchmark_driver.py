"""Validate explicit benchmark paths and prepare the notebook's two-run plan.

This slice does not launch model servers, inference workers, or GPU telemetry.
"""

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_PAGES = 1651
PREFLIGHT_INDICES = [0, 103, 309, 584, 825, 1066, 1341, 1547, 1650]
WARMUP_INDICES = [34, 859, 1616]
CONCURRENCIES = [1, 24]
VARIANTS = {
    "BF16": ("BF16_W16A16", 16),
    "W8A16": ("GPTQ_W8A16_G128", 8),
    "W4A16": ("GPTQ_W4A16_G128", 4),
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def nested_values(value, key):
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key == key:
                yield child
            yield from nested_values(child, key)
    elif isinstance(value, list):
        for child in value:
            yield from nested_values(child, key)


def require_file(path):
    if not path.is_file():
        raise ValueError(f"Required file missing: {path}")
    return path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_run(bundle, model, output, variant, mode):
    """Return a validated, JSON-serializable plan without creating output."""
    bundle, model, output = (Path(path).resolve() for path in (bundle, model, output))
    if variant not in VARIANTS or mode not in ("PREFLIGHT", "FULL"):
        raise ValueError("Variant must be BF16/W8A16/W4A16; mode PREFLIGHT/FULL")
    if output.exists():
        raise ValueError(f"Output path must be new: {output}")
    manifest = load_json(require_file(bundle / "bundle_manifest.json"))
    if manifest.get("bundle") != "paddleocr_vl_1_6_l4_wna16_bundle_v1":
        raise ValueError("Bundle identity mismatch")
    paths = manifest["paths"]
    inputs = load_json(require_file(bundle / paths["benchmark_inputs"]))
    truth = load_json(
        require_file(bundle / paths["omnidocbench"] / "OmniDocBench.json")
    )
    if (
        not isinstance(inputs, list)
        or not isinstance(truth, list)
        or len(inputs) != EXPECTED_PAGES
        or len(truth) != EXPECTED_PAGES
    ):
        raise ValueError("OmniDocBench page-count drift")
    layout = bundle / paths["layout"]
    for name in ("inference.json", "inference.pdiparams", "inference.yml"):
        require_file(layout / name)
    require_file(model / "config.json")
    weights = sorted(model.glob("*.safetensors"))
    if not weights or any(
        not path.is_file() or path.stat().st_size == 0 for path in weights
    ):
        raise ValueError(f"Checkpoint has no valid safetensors: {model}")
    config = load_json(model / "config.json")
    bits = VARIANTS[variant][1]
    if bits != 16:
        audit = load_json(require_file(model / "scope_audit.json"))
        required = {
            "algorithm": "GPTQ",
            "group_size": 128,
            "activation_dtype": "BF16",
            "calibration_count": 512,
            "target_count": 234,
            "vision_layer_indices": list(range(27)),
            "decoder_layer_indices": list(range(18)),
        }
        for key, expected in required.items():
            if audit.get(key) != expected:
                raise ValueError(f"Checkpoint audit mismatch for {key}")
        quant = config.get("quantization_config")
        if not isinstance(quant, dict):
            raise ValueError("Missing quantization_config")
        if bits not in {
            int(x) for x in nested_values(quant, "num_bits") if x is not None
        } or 128 not in {
            int(x) for x in nested_values(quant, "group_size") if x is not None
        }:
            raise ValueError("Quantization config mismatch")
    page_ids = []
    for position, entry in enumerate(inputs):
        index = int(entry.get("annotation_index", position))
        if not 0 <= index < EXPECTED_PAGES:
            raise ValueError(f"Invalid annotation index at row {position}")
        page_id = entry["page_id"]
        image = bundle / entry["image"]
        if not image.is_file():
            raise ValueError(f"Benchmark image missing: {image}")
        image_ref = truth[index].get("page_info", {}).get("image_path")
        if (
            not isinstance(image_ref, str)
            or not image_ref
            or Path(image_ref.replace("\\", "/")).stem != page_id
        ):
            raise ValueError(f"Prediction-name mismatch at {index}")
        page_ids.append(page_id)
    if len(set(page_ids)) != EXPECTED_PAGES:
        raise ValueError("OmniDocBench page IDs are not unique")
    measured = PREFLIGHT_INDICES if mode == "PREFLIGHT" else list(range(EXPECTED_PAGES))
    file_manifest = []
    for path in (
        model / "config.json",
        *weights,
        *([model / "scope_audit.json"] if bits != 16 else []),
    ):
        file_manifest.append(
            {
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "variant_key": variant,
        "variant_id": VARIANTS[variant][0],
        "checkpoint": str(model),
        "checkpoint_files": file_manifest,
        "bundle": str(bundle),
        "layout": str(layout),
        "output_root": str(output),
        "run_mode": mode,
        "concurrencies": CONCURRENCIES.copy(),
        "measured_indices": list(measured),
        "warmup_indices": WARMUP_INDICES.copy(),
        "expected_prediction_filenames": [page_ids[i] + ".md" for i in measured],
        "runs": [
            {"concurrency": n, "output_dir": str(output / f"c{n}")}
            for n in CONCURRENCIES
        ],
    }


def run_plan(plan, start_server, run_client, stop_server):
    """Drive independent c1 then c24 runs; callers supply the GPU runtime.

    The CLI deliberately does not invoke this hook until server/client/telemetry
    integration has been tested on the authorized original topology.
    """
    results = []
    for run in plan["runs"]:
        server = None
        try:
            server = start_server(run)
            results.append(run_client(run, server))
        finally:
            if server is not None:
                stop_server(server)
    if [result["request_concurrency"] for result in results] != CONCURRENCIES:
        raise ValueError("Completed concurrency order drift")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory; not created by this command",
    )
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--mode", choices=("PREFLIGHT", "FULL"), required=True)
    parser.add_argument(
        "--write-plan",
        action="store_true",
        help="Create output/benchmark_plan.json after validation; no inference",
    )
    args = parser.parse_args()
    try:
        plan = prepare_run(
            args.bundle, args.model, args.output, args.variant, args.mode
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    if args.write_plan:
        try:
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output / "benchmark_plan.json").write_text(
                json.dumps(plan, indent=2) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            parser.error(str(exc))
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
