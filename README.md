# PaddleOCR-VL-1.6 weight-only GPTQ capstone

Code and compact, checkable evidence for the BF16, GPTQ-W8A16-G128 and GPTQ-W4A16-G128 document-parsing study. This repository contains **code and evidence only**; it does not contain the submitted paper, black book, model weights, third-party images, or offline dependency wheelhouses. The two GPTQ Hugging Face model repositories are **separate releases**; this GitHub repository's visibility does not control theirs. The submitted documents are not assumed to be older local drafts.

## What is here

- `scripts/`, `tests/`, `config/`: tested portable checkpoint, serving, parsing, benchmarking and evaluator helpers with pinned identity files.
- `notebooks/`: source-only portable/Kaggle-oriented reference workflows, with source hashes; saved outputs are separate evidence, not silently recreated.
- `evidence/official_scores/`: six official metric-result JSONs (BF16/W8A16/W4A16 at c=1 and c=24) and c=24 quality summaries/job status.
- `evidence/performance_recomputed.json`: six full-run rows recomputed from archived page records.
- `evidence/smoke/`: one-page synthetic functional GPU smoke evidence for each low-bit model, plus Marlin-selection log excerpt. Smoke is not an accuracy or performance benchmark.
- `evidence/full_run_archive_hashes.json`: SHA-256/size for the three complete local full-run ZIPs. Large archives are deliberately not in GitHub.
- `evidence/laptop_smoke_20261030/`: real RTX 4070 Laptop sequential BF16/W8A16/W4A16 output, timings, device-memory readback and full browser recording on one clearly labeled synthetic page. This is not a benchmark, accuracy result or examiner-page claim.
- `evidence/presentation_demo_20261030/`: **48-second PPT-themed captioned edit** of a new, real three-model laptop run. Long unchanged loading waits are shortened; source frames and logs remain local. The input is a labeled synthetic page, not an examiner page.
- `demo/`: localhost-only selectable, sequential three-variant UI with a new-page preview and clear unsupported-file recovery. PNG/JPEG and one-page PDF are supported; other file types and multi-page PDFs are rejected before GPU work. A one-page PDF and JPEG were exercised locally, but arbitrary examiner pages remain unverified. See `demo/README_DEMO.md`.
- `docs/REPRODUCTION.md`: exact scope, notebook order, CLI commands and known reproduction limits.

The experiment used all 1,651 OmniDocBench v1.6 pages per variant at one and 24 within-page concurrent region requests, under the original four-L4 notebook topology. Throughput runs occurred on different machines, so a one-GPU laptop demonstration is **not** a benchmark replication.

## Models and license boundary

See `models/README.md`. The original reproduction/demo code and source notebooks in this repository are licensed under **Apache License 2.0** (`LICENSE` and `NOTICE`); the upstream PaddleOCR-VL-1.6 model and separately hosted GPTQ checkpoints also carry Apache-2.0 with their own attribution. Third-party benchmark/calibration data and offline package bundles are excluded; this license does **not** grant rights to those datasets or tools.

## Tests

Run `PYTHONPATH=scripts python -m unittest discover -s tests -v` for portable unit tests. This does not run GPU inference or the official evaluator. The separate local evidence folder is `/home/Master/Documents/Hermes/capstone-evidence`; three copied raw archives were hash-checked against their source manifests, and full browser capture frames/logs are retained there. Run `cd demo && PYTHONPATH=. python -m unittest discover -s tests -q` for its 56 device-free tests; the portable suite has 29. These tests do not substitute for the recorded laptop GPU run.
