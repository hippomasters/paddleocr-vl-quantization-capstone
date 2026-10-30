# PaddleOCR-VL-1.6 GPTQ: inference and benchmark reproduction

This repository contains **code and small provenance files only**. It does not redistribute model weights, benchmark images, calibration pages, cached third-party wheels, or the author's original full-run ZIP archives. Obtain the upstream data/models under their own terms. W8A16 and W4A16 checkpoints belong in separate model repositories.

## What is verified

- The checkpoint weight, config, and scope-audit files match their full-run SHA-256 records; all 14 shipped files per quantized model (including custom Python code) have a path-free checksum manifest.
- `serve_model.py` constructs the vLLM flags recorded in the full benchmark; `parse_page.py` uses the recorded PaddleOCR-VL-1.6 layout/prediction options. Their CLI and configuration behavior are unit-tested, **not GPU-tested in this package**.
- `evaluate_official.py --dry-run` validated all three real 1,651-page c=24 prediction directories, including the one shared empty prediction. Its Docker command uses the **same pinned evaluator image digest and metric definitions** as the c=1 study; worker counts are explicit options (the original low-bit c=1 evaluations used match/TEDS/CDM workers 4/4/2).
- `summarize_performance.py` was executed on all three full-run archives; its `verified_results/performance_recomputed.json` reproduces the recorded throughput and paired speedups.
- The three `notebooks/` files are the original study source cells with bulky saved outputs and execution metadata removed. Their code was not changed; source digests are in `notebooks/source_manifest.json`. They remain **Kaggle-oriented reference workflows**, not platform-independent one-click scripts.

## Use an existing checkpoint on page images

Use **separate Python 3.12 environments/processes** for vLLM serving and the PaddleOCR client: the original quantization and serving dependencies are intentionally isolated. `config/environment_versions.json` records what produced the benchmark. On a GPU with less memory than the study's L4, lower the server capacity settings; that is a usage configuration, **not** a reproduction of the paper's throughput.

1. Obtain a W8A16 or W4A16 model directory from its model repository and the upstream PP-DocLayoutV3 directory at the pinned revision in `config/source_revisions.json`.
2. Verify the complete model snapshot, including custom Python code: `python scripts/verify_checkpoints.py --manifest config/checkpoint_hashes.json --variant W8A16 --checkpoint MODEL_DIR`. Review that code before trusting it.
3. In the vLLM environment, run `python scripts/serve_model.py --model MODEL_DIR --served-model-name GPTQ_W8A16_G128 --variant W8A16 --port 8000`. The launcher **rechecks all 14 original files before enabling custom model code** and binds the endpoint to the local machine. Use `--variant W4A16` for the 4-bit checkpoint.
4. In the PaddleOCR environment, run `python scripts/parse_page.py --image PAGE_IMAGE --layout-dir LAYOUT_DIR --server-url SERVER_URL --served-model-name GPTQ_W8A16_G128 --output-dir NEW_OUTPUT_DIR`. Set `SERVER_URL` to your local vLLM API base (including `/v1`). This writes one Markdown file and one structured JSON file; use the W4A16 name for that checkpoint.

For the study’s two-GPU topology, keep the server on physical GPU 0 and run the PaddleOCR client with only physical GPU 1 visible (the client’s `gpu:0` then means its own first visible GPU). One-GPU serving and layout together are not validated here.

The full document pipeline is **PP-DocLayoutV3 layout → region crops/task prompts → PaddleOCR-VL-1.6 VLM via vLLM → Markdown**. A single model checkpoint is only the VLM stage.

## Reproduce the original experiment

The exact study used four NVIDIA L4 GPUs (GPU 0 for vLLM, GPU 1 for layout, the others unused), Python 3.12, pinned model/dataset revisions, offline inference, and one page at a time with within-page request concurrency 1 or 24. Do not compare throughput from other hardware as a controlled replication.

1. `notebooks/01_build_bundle.ipynb`: fetch pinned BF16/layout models, OmniDocBench v1.6, 512 disjoint DocLayNet-base calibration pages, and separate offline wheelhouses. It requires internet access and disk space; review upstream dataset/model terms.
2. `notebooks/02_quantize_checkpoints.ipynb`: attach the builder output, keep inference internet disabled, and create/audit W8A16 and W4A16. This notebook applies compatibility patches to the original quantization worker; **its effective patched worker** is the method used by the saved checkpoints.
3. `notebooks/03_full_benchmark.ipynb`: attach the builder output and checkpoints, set `SELECTED_VARIANT`, run its nine-page PREFLIGHT, then set FULL and rerun from the beginning. Repeat for BF16, W8A16, W4A16; each full run creates c1 and c24 predictions plus per-page/request logs and telemetry. This notebook's provided copy is source-only; the separate archived runs are the evidence for the paper's measurements.
4. Check file identity with `python scripts/verify_checkpoints.py --manifest config/checkpoint_hashes.json --variant W8A16 --checkpoint MODEL_DIR` (repeat for W4A16).
5. Recompute performance from the three result ZIPs with `python scripts/summarize_performance.py --bf16 BF16_ZIP --w8a16 W8_ZIP --w4a16 W4_ZIP --output NEW_SUMMARY.json`.
6. Extract each run's `c1/evaluator_input` or `c24/evaluator_input` into its own **flat** prediction directory, use the official OmniDocBench ground-truth JSON, and run `python scripts/evaluate_official.py --ground-truth GT_JSON --predictions PRED_DIR --output-dir NEW_EVAL_DIR`. Use `--dry-run` first. The Docker image is pinned by digest; the output directory must not exist beforehand. The evaluator preserves and scores any empty Markdown prediction rather than silently replacing it.

Do not upload the full offline bundle: it contains third-party datasets and large dependencies. Release rights and a license for **this reproduction code** must be settled by the owner before public publication. A clean end-to-end run from a newly published Hub download remains a release gate.

## Portable benchmark preflight slice (CPU only)

`benchmark_driver.py` accepts explicit paths, not `/kaggle/input` or `/kaggle/working` assumptions. Point `--bundle` at the **inner** `paddleocr_vl_1_6_l4_wna16_bundle` directory containing `bundle_manifest.json`; use an independently attached checkpoint directory as `--model`. The builder manifest records the BF16 directory as `models/PaddleOCR-VL-1.6-BF16` and the quantized output directory names as `PaddleOCR-VL-1.6-GPTQ-W8A16-G128` and `PaddleOCR-VL-1.6-GPTQ-W4A16-G128`. Kaggle attachment slugs are not fixed by the notebook: locate these directories under your mounted inputs, then pass their actual paths. No dependency rebuilding or re-quantization is required.

```sh
python scripts/benchmark_driver.py --bundle BUNDLE_DIR --model MODEL_DIR \
  --output NEW_OUTPUT_DIR --variant W8A16 --mode PREFLIGHT --write-plan
```

This checks bundle identity, all 1,651 image/ground-truth name pairs, layout files, checkpoint weights/configuration and quantized scope audit, and streams SHA-256 of model files. It writes only `benchmark_plan.json` into a new output directory. The plan fixes the nine notebook preflight indices and warmups, or all 1,651 measured pages for `--mode FULL`; both list **c1 then c24** with separate output directories. Omit `--write-plan` to inspect the JSON without creating files. The module also exposes a callback-driven `run_plan` lifecycle that starts and stops independent servers in order, but **the CLI does not launch inference, telemetry, install wheelhouses, validate L4 hardware, or produce a PREFLIGHT: PASS**. A FULL plan is not a FULL execution and must not be treated as permission to skip the notebook's successful preflight. Full original-topology inference remains blocked until authorized four-L4-equivalent access; do not use restricted competition GPUs for this unrelated work.

## Local tests

Run `PYTHONPATH=scripts python -m unittest discover -s tests -v`. The unit tests need only Python's standard library. The Docker evaluator and GPU inference require their own runtimes and are deliberately not invoked by the test suite.
