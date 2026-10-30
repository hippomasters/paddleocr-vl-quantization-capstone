"""Reconstruct paper throughput and reliability from archived page records."""

import json
from pathlib import Path
import zipfile


def load_run(archive: Path, concurrency: str, *, expected_pages: int = 1651) -> dict:
    if concurrency not in ("c1", "c24"):
        raise ValueError("Concurrency must be c1 or c24")
    with zipfile.ZipFile(archive) as z:
        rows = [
            json.loads(line)
            for line in z.read(f"{concurrency}/page_metrics.jsonl").splitlines()
            if line.strip()
        ]
    pages = {}
    for row in rows:
        page_id = row["page_id"]
        if page_id in pages:
            raise ValueError(f"Duplicate page ID: {page_id}")
        import math

        page_time = float(row["page_inference_seconds"])
        vlm_time = float(row["vlm_seconds"])
        if not math.isfinite(page_time) or not math.isfinite(vlm_time):
            raise ValueError(f"Nonfinite timing value: {page_id}")
        if page_time <= 0 or vlm_time < 0:
            raise ValueError(f"Invalid page or VLM time: {page_id}")
        if int(row["generated_tokens"]) < 0:
            raise ValueError(f"Negative generated-token count: {page_id}")
        pages[page_id] = row
    if len(pages) != expected_pages:
        raise ValueError(f"Expected {expected_pages} pages, found {len(pages)}")
    return {
        "page_count": len(pages),
        "pages": pages,
        "total_page_seconds": sum(
            float(x["page_inference_seconds"]) for x in pages.values()
        ),
        "total_vlm_seconds": sum(float(x["vlm_seconds"]) for x in pages.values()),
        "total_tokens": sum(int(x["generated_tokens"]) for x in pages.values()),
        "token_limit_pages": sum(
            bool(x["token_limit_reached"]) for x in pages.values()
        ),
        "completed_pages": sum(x["status"] == "completed" for x in pages.values()),
    }


def percentile(values, percentage: float) -> float:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        raise ValueError("No page times")
    pos = (len(ordered) - 1) * percentage / 100
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] * (1 - (pos - lower)) + ordered[upper] * (pos - lower)


def summarize_run(run: dict) -> dict:
    n = run["page_count"]
    elapsed = run["total_page_seconds"]
    vlm = run["total_vlm_seconds"]
    return {
        "page_count": n,
        "completed_pages": run["completed_pages"],
        "mean_page_seconds": elapsed / n,
        "p95_page_seconds": percentile(
            (p["page_inference_seconds"] for p in run["pages"].values()), 95
        ),
        "pages_per_hour": 3600 * n / elapsed,
        "vlm_tokens_per_second": run["total_tokens"] / vlm if vlm else None,
        "generated_tokens": run["total_tokens"],
        "token_limit_pages": run["token_limit_pages"],
    }


def paired_speedup(reference: dict, candidate: dict) -> float:
    import math

    if set(reference["pages"]) != set(candidate["pages"]):
        raise ValueError("Paired speedup requires identical page IDs")
    ratios = (
        math.log(
            reference["pages"][key]["page_inference_seconds"]
            / candidate["pages"][key]["page_inference_seconds"]
        )
        for key in reference["pages"]
    )
    return math.exp(sum(ratios) / reference["page_count"])


def validate_common_cohort(runs: dict) -> None:
    cohorts = {
        frozenset(run["pages"])
        for variants in runs.values()
        for run in variants.values()
    }
    if len(cohorts) != 1:
        raise ValueError("All conditions must use the same page cohort")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--w8a16", type=Path, required=True)
    parser.add_argument("--w4a16", type=Path, required=True)
    parser.add_argument("--expected-pages", type=int, default=1651)
    parser.add_argument("--output", type=Path, help="Write a new JSON summary file")
    args = parser.parse_args()
    if args.expected_pages < 1 or (args.output and args.output.exists()):
        parser.error("Page count must be positive and output must not exist")
    paths = {"BF16": args.bf16, "W8A16": args.w8a16, "W4A16": args.w4a16}
    runs = {}
    for variant, path in paths.items():
        with zipfile.ZipFile(path) as z:
            contract = json.loads(z.read("selected_contract.json"))
        if contract["variant_key"] != variant:
            parser.error(f"{path} is not a {variant} archive")
        runs[variant] = {
            c: load_run(path, c, expected_pages=args.expected_pages)
            for c in ["c1", "c24"]
        }
    validate_common_cohort(runs)
    result = {"expected_pages": args.expected_pages, "variants": {}}
    for variant in paths:
        result["variants"][variant] = {}
        for c in ["c1", "c24"]:
            entry = summarize_run(runs[variant][c])
            entry["paired_speedup_vs_bf16"] = (
                1.0
                if variant == "BF16"
                else paired_speedup(runs["BF16"][c], runs[variant][c])
            )
            result["variants"][variant][c] = entry
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
