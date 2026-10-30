# PaddleOCR-VL-1.6 weight-only GPTQ capstone

Code and compact, checkable evidence for the BF16, GPTQ-W8A16-G128 and GPTQ-W4A16-G128 document-parsing study. This is a **private staging repository**; it does not contain the submitted paper, black book, model weights, third-party images, or offline dependency wheelhouses. The submitted documents are not assumed to be the older local drafts.

## What is here

- `scripts/`, `tests/`, `config/`: tested portable checkpoint, serving, parsing, benchmarking and evaluator helpers with pinned identity files.
- `notebooks/`: source-only portable/Kaggle-oriented reference workflows, with source hashes; saved outputs are separate evidence, not silently recreated.
- `evidence/official_scores/`: six official metric-result JSONs (BF16/W8A16/W4A16 at c=1 and c=24) and c=24 quality summaries/job status.
- `evidence/performance_recomputed.json`: six full-run rows recomputed from archived page records.
- `evidence/smoke/`: one-page synthetic functional GPU smoke evidence for each low-bit model, plus Marlin-selection log excerpt. Smoke is not an accuracy or performance benchmark.
- `evidence/full_run_archive_hashes.json`: SHA-256/size for the three complete local full-run ZIPs. Large archives are deliberately not in GitHub.
- `evidence/laptop_smoke_20261030/`: real RTX 4070 Laptop sequential BF16/W8A16/W4A16 output, timings, device-memory readback and full browser recording on one clearly labeled synthetic page. This is not a benchmark, accuracy result or examiner-page claim.
- `demo/`: localhost-only selectable, sequential three-variant UI and 52 CPU-tested orchestration/validation tests. See `demo/README_DEMO.md`. A real three-model laptop GPU run and synthetic-page browser recording are under `evidence/laptop_smoke_20261030/`; a one-page PDF and JPEG were also exercised locally, but arbitrary examiner pages remain unverified.
- `docs/REPRODUCTION.md`: exact scope, notebook order, CLI commands and known reproduction limits.

The experiment used all 1,651 OmniDocBench v1.6 pages per variant at one and 24 within-page concurrent region requests, under the original four-L4 notebook topology. Throughput runs occurred on different machines, so a one-GPU laptop demonstration is **not** a benchmark replication.

## Models and license boundary

See `models/README.md`. The upstream PaddleOCR-VL-1.6 model and redistributed GPTQ checkpoints carry the upstream Apache-2.0 license. **This repository's original code has not yet been assigned an open-source license.** Do not infer that third-party benchmark/calibration data may be redistributed from its absence here. Before changing this repository or the HF models to public, confirm code licensing, data rights, and a clean model download-to-inference test.

## Tests

Run `PYTHONPATH=scripts python -m unittest discover -s tests -v` for portable unit tests. This does not run GPU inference or the official evaluator. The full raw records, third-party inputs and private weights are kept in a separate local evidence folder and model repositories. The local folder is `/home/Master/Documents/Hermes/capstone-evidence`; three copied raw archives were hash-checked against their source manifests. The folder and its video capture frames are not tracked by Git. Run `cd demo && PYTHONPATH=. python -m unittest discover -s tests -q` for its 52 device-free tests.
