"""Score OmniDocBench predictions with the study's pinned Docker evaluator."""

from pathlib import Path

IMAGE = (
    "ghcr.io/zeng-weijun/omnidocbench-eval@sha256:"
    "6116ad72172e763b5c43e963d5efebf2093f2362b975f58156ce4f6c9142e617"
)


def build_docker_command(
    ground_truth: Path, predictions: Path, results: Path, config: Path, name: str
) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        "none",
        "--shm-size",
        "4g",
        "--entrypoint",
        "bash",
        "-v",
        f"{Path(ground_truth).resolve()}:/workspace/gt/OmniDocBench.json:ro",
        "-v",
        f"{Path(predictions).resolve()}:/workspace/data_md/predictions:ro",
        "-v",
        f"{Path(results).resolve()}:/workspace/result",
        "-v",
        f"{Path(config).resolve()}:/tmp/eval.yaml:ro",
        IMAGE,
        "-lc",
        (
            "cd /workspace && PYTHONUNBUFFERED=1 python -u "
            "pdf_validation.py --config /tmp/eval.yaml"
        ),
    ]


def validate_inputs(
    ground_truth: Path, predictions: Path, *, expected_pages: int = 1651
) -> dict:
    import json

    truth = json.loads(Path(ground_truth).read_text(encoding="utf-8"))
    if not isinstance(truth, list) or len(truth) != expected_pages:
        raise ValueError(f"Expected {expected_pages} ground-truth pages")
    expected = set()
    for row in truth:
        image = row["page_info"]["image_path"].replace("\\", "/").rsplit("/", 1)[-1]
        filename = Path(image).stem + ".md"
        if filename in expected:
            raise ValueError(f"Duplicate prediction filename: {filename}")
        expected.add(filename)
    folder = Path(predictions)
    if not folder.is_dir() or any(
        p.is_dir() or p.is_symlink() for p in folder.iterdir()
    ):
        raise ValueError("Predictions must be a flat directory of regular files")
    found = {p.name for p in folder.iterdir() if p.is_file() and p.suffix == ".md"}
    if found != expected:
        raise ValueError(
            (
                f"Prediction alignment failed: missing={len(expected - found)}, "
                f"extra={len(found - expected)}"
            )
        )
    empty = sorted(
        p.name for p in folder.iterdir() if p.suffix == ".md" and p.stat().st_size == 0
    )
    return {"prediction_files": len(found), "empty_filenames": empty}


def validate_output_location(predictions: Path, output_dir: Path) -> None:
    source = Path(predictions).resolve()
    destination = Path(output_dir).resolve()
    if destination == source or destination.is_relative_to(source):
        raise ValueError("Evaluator output cannot be inside predictions")


def main() -> int:
    import argparse
    import hashlib
    import json
    import subprocess
    import sys
    from uuid import uuid4

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--match-workers", type=int, default=1)
    parser.add_argument("--teds-workers", type=int, default=1)
    parser.add_argument("--cdm-workers", type=int, default=1)
    parser.add_argument("--expected-pages", type=int, default=1651)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (
        min(
            args.match_workers, args.teds_workers, args.cdm_workers, args.expected_pages
        )
        < 1
    ):
        parser.error("workers and page count must be positive")
    if args.output_dir.exists():
        parser.error("output directory already exists; refusing overwrite")
    try:
        validate_output_location(args.predictions, args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        check = validate_inputs(
            args.ground_truth, args.predictions, expected_pages=args.expected_pages
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps({**check, "image": IMAGE, "dry_run": True}, indent=2))
        return 0
    args.output_dir.mkdir(parents=True)
    config = args.output_dir / "eval_config.yaml"
    config.write_text(
        f"""end2end_eval:
  metrics:
    text_block:
      metric: [Edit_dist]
    display_formula:
      metric: [Edit_dist, CDM]
      cdm_workers: {args.cdm_workers}
    table:
      metric: [TEDS, Edit_dist]
      teds_workers: {args.teds_workers}
    reading_order:
      metric: [Edit_dist]
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: /workspace/gt/OmniDocBench.json
    prediction:
      data_path: /workspace/data_md/predictions
    match_method: quick_match
    match_workers: {args.match_workers}
    quick_match_truncated_timeout_sec: 300
    match_timeout_sec: 420
    timeout_fallback_max_chunk_span: 10
    timeout_fallback_order_penalty: 0.10
""",
        encoding="utf-8",
    )
    digest = hashlib.sha256(args.ground_truth.read_bytes()).hexdigest()
    (args.output_dir / "input_validation.json").write_text(
        json.dumps({**check, "ground_truth_sha256": digest, "image": IMAGE}, indent=2)
    )
    command = build_docker_command(
        args.ground_truth,
        args.predictions,
        args.output_dir,
        config,
        "omnidocbench-" + uuid4().hex[:12],
    )
    with (args.output_dir / "evaluation.log").open("w", encoding="utf-8") as log:
        run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if run.returncode:
        print(
            (
                f"Docker evaluator failed (exit {run.returncode}); see "
                f"{args.output_dir / 'evaluation.log'}"
            ),
            file=sys.stderr,
        )
        return run.returncode
    metric = args.output_dir / "predictions_quick_match_metric_result.json"
    if not metric.is_file():
        print("Docker completed without an official metric file", file=sys.stderr)
        return 2
    m = json.loads(metric.read_text())
    if m["match_debug"]["page_count"] != args.expected_pages:
        print("Official evaluator page count differs from expected", file=sys.stderr)
        return 2
    text = 100 * (1 - m["text_block"]["page"]["Edit_dist"]["ALL"])
    tables = 100 * m["table"]["page"]["TEDS"]["ALL"]
    formulas = 100 * m["display_formula"]["page"]["CDM"]["ALL"]
    order = 100 * (1 - m["reading_order"]["page"]["Edit_dist"]["ALL"])
    score = {
        "overall": (text + tables + formulas) / 3,
        "text": text,
        "tables": tables,
        "formulas": formulas,
        "reading_order": order,
        "page_count": args.expected_pages,
        "empty_predictions": check["empty_filenames"],
    }
    (args.output_dir / "quality_summary.json").write_text(json.dumps(score, indent=2))
    print(json.dumps(score, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
