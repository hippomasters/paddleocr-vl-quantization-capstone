"""Parse one document page using PaddleOCR-VL-1.6 and a vLLM checkpoint server."""

from pathlib import Path

PREDICT_KWARGS = {
    "min_pixels": 112896,
    "max_pixels": 1003520,
    "max_new_tokens": 4096,
    "repetition_penalty": 1.0,
    "temperature": 0.0,
    "top_p": 1.0,
}


def build_pipeline_kwargs(
    layout_dir: Path, server_url: str, served_name: str, concurrency: int, device: str
) -> dict:
    return {
        "pipeline_version": "v1.6",
        "device": device,
        "layout_detection_model_name": "PP-DocLayoutV3",
        "layout_detection_model_dir": str(layout_dir),
        "vl_rec_backend": "vllm-server",
        "vl_rec_server_url": server_url,
        "vl_rec_api_model_name": served_name,
        "vl_rec_max_concurrency": concurrency,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_layout_detection": True,
        "use_chart_recognition": False,
        "use_seal_recognition": False,
        "use_ocr_for_image_block": False,
        "use_queues": True,
        "format_block_content": False,
        "merge_layout_blocks": True,
        "markdown_ignore_labels": [],
        "enable_hpi": False,
        "use_tensorrt": False,
        "enable_cinn": False,
    }


def parse_one_image(pipeline, image: Path, output_dir: Path) -> dict:
    import json

    image = Path(image)
    output_dir = Path(output_dir)
    if not image.is_file():
        raise FileNotFoundError(image)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    predictions = list(pipeline.predict(str(image), **PREDICT_KWARGS))
    if len(predictions) != 1:
        raise RuntimeError(f"Expected one page prediction, got {len(predictions)}")
    output_dir.mkdir(parents=True)
    markdown = output_dir / f"{image.stem}.md"
    structured = output_dir / f"{image.stem}.json"
    predictions[0].save_to_markdown(
        str(markdown), pretty=False, show_formula_number=False
    )
    predictions[0].save_to_json(str(structured))
    if not markdown.is_file() or not structured.is_file():
        raise RuntimeError("Parser failed to write Markdown and JSON outputs")
    json.loads(structured.read_text(encoding="utf-8"))
    return {"markdown": str(markdown), "json": str(structured)}


def main() -> int:
    import argparse
    import json
    from urllib.parse import urlparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="A single page image")
    parser.add_argument(
        "--layout-dir", type=Path, required=True, help="PP-DocLayoutV3 model directory"
    )
    parser.add_argument(
        "--server-url", required=True, help="vLLM API base URL, including /v1"
    )
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New output directory"
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--device", default="gpu:0", help="Paddle layout device")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Page image not found: {args.image}")
    required = ["inference.json", "inference.pdiparams", "inference.yml"]
    missing = [name for name in required if not (args.layout_dir / name).is_file()]
    if missing:
        parser.error(f"Layout directory missing: {missing}")
    if args.output_dir.exists() or args.concurrency < 1:
        parser.error("output directory must be new and concurrency must be positive")
    parsed = urlparse(args.server_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        parser.error("server URL must have an HTTP(S) scheme and host")
    try:
        import paddle
        from paddleocr import PaddleOCRVL
    except ImportError as exc:
        parser.error(f"Install the pinned PaddleOCR client runtime: {exc}")
    paddle.seed(2026)
    pipeline = PaddleOCRVL(
        **build_pipeline_kwargs(
            args.layout_dir,
            args.server_url,
            args.served_model_name,
            args.concurrency,
            args.device,
        )
    )
    with paddle.no_grad():
        outputs = parse_one_image(pipeline, args.image, args.output_dir)
    if args.device.startswith("gpu"):
        paddle.device.synchronize()
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
