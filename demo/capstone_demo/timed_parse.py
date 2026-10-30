"""Run parse_page.py's tested pipeline on one page and record per-pass latency.

Runs inside the PaddleOCR client environment. It reuses parse_page's
build_pipeline_kwargs / parse_one_image / PREDICT_KWARGS and the same
paddle.seed + PaddleOCRVL construction as parse_page.main; only timing is added.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlparse

SOURCE_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SOURCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SOURCE_SCRIPTS))

from parse_page import build_pipeline_kwargs, parse_one_image  # noqa: E402

LAYOUT_FILES = ("inference.json", "inference.pdiparams", "inference.yml")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--layout-dir", type=Path, required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timing-out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--variant-label", required=True)
    parser.add_argument("--warmup-passes", type=int, choices=[0, 1, 2], default=1)
    return parser


def main(argv=None, *, paddle_module=None, pipeline_factory=None, clock=time.perf_counter):
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.image.is_file():
        parser.error(f"Page file not found: {args.image}")
    missing = [n for n in LAYOUT_FILES if not (args.layout_dir / n).is_file()]
    if missing:
        parser.error(f"Layout directory missing: {missing}")
    url = urlparse(args.server_url)
    if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost"):
        parser.error("server URL must be a local http://127.0.0.1 endpoint")
    warmup_dirs = [
        args.output_dir.with_name(f"{args.output_dir.name}.warmup{i + 1}")
        for i in range(args.warmup_passes)
    ]
    for path in [args.output_dir, args.timing_out, *warmup_dirs]:
        if path.exists():
            parser.error(f"Refusing to overwrite {path}")
    if paddle_module is None or pipeline_factory is None:
        try:
            import paddle
            from paddleocr import PaddleOCRVL
        except ImportError as exc:
            parser.error(f"Install the pinned PaddleOCR client runtime: {exc}")
        paddle_module, pipeline_factory = paddle, PaddleOCRVL
    paddle_module.seed(2026)
    kwargs = build_pipeline_kwargs(
        args.layout_dir, args.server_url, args.served_model_name, 1, args.device
    )
    start = clock()
    pipeline = pipeline_factory(**kwargs)
    pipeline_init_s = clock() - start
    passes = []
    targets = [("warmup", d) for d in warmup_dirs] + [("measured", args.output_dir)]
    outputs = None
    for kind, directory in targets:
        with paddle_module.no_grad():
            began = clock()
            outputs = parse_one_image(pipeline, args.image, directory)
            if args.device.startswith("gpu"):
                paddle_module.device.synchronize()
            passes.append({"kind": kind, "seconds": clock() - began})
    timing = {
        "variant": args.variant_label,
        "served_model_name": args.served_model_name,
        "layout_device": args.device,
        "concurrency": 1,
        "pipeline_init_s": pipeline_init_s,
        "passes": passes,
        "measured_parse_s": passes[-1]["seconds"],
        "scope": "parse_one_image wall time: layout + VLM requests + Markdown/JSON save",
        "outputs": outputs,
    }
    with args.timing_out.open("x", encoding="utf-8") as stream:
        json.dump(timing, stream, indent=2)
    print(json.dumps(timing, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
