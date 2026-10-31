# Release validation

The first section records the original private-model staging audit. The laptop update below
supersedes its earlier "clean download pending" gate. Repository visibility is checked
independently for GitHub code and Hugging Face models.

## Verified

- Original BF16, W8A16, and W4A16 checkpoint directories are present. For each, `config.json` and `model.safetensors` match the archived benchmark contract by exact size and SHA-256; both GPTQ `scope_audit.json` files also match. The original quantization target lists agree (234 matrices, group size 128, calibration count 512).
- Safetensors headers/offsets, JSON syntax, and bundled Python syntax were checked. Each quantized model's `AutoConfig` and `AutoTokenizer` loaded **offline in an isolated Python 3.12 environment with Transformers 5.17.0**. This does not exercise its VLM weights or image processor at inference time.
- `reproduction/scripts/verify_checkpoints.py` verified the real W8A16 and W4A16 release candidates against `config/checkpoint_hashes.json`, including all 14 files and custom Python code; unlisted executables are rejected.
- `reproduction/scripts/evaluate_official.py --dry-run` accepted the **three real 1,651-page c24** prediction directories, retaining the shared empty page; no additional Docker evaluation was started by this script.
- `reproduction/scripts/summarize_performance.py` recomputed six full-run performance rows from individual page records and wrote `reproduction/verified_results/performance_recomputed.json`.
- Independent review findings were addressed: all model code is hashed before `trust_remote_code`, unknown executables are refused, evaluator outputs cannot overlap prediction inputs, all six performance rows must share a cohort, and non-finite timing values are rejected. The safe and rejected evaluator dry-run paths were both exercised against real predictions.
- The portable reproduction unit suite passed **29 tests** with Ruff `E,F,B`; the separate GPU smoke package passed **12 CPU tests** with Ruff `E,F,B`.


- All three official c24 Docker evaluations completed with exit code 0 and 1,651 pages each; their score summaries and logs are stored in `../omnidocbench_c24_official_20261013T030825Z/`. Overall scores: BF16 96.37532135098094, W8A16 96.34418380849793, W4A16 93.74095470090283.
- W8A16 passed **one synthetic-page functional GPU smoke test** on the user-run Kaggle session: hashes for 14 model files and the executed dependency bundle verified; GPU 0 served `GPTQ_W8A16_G128`, GPU 1 ran PP-DocLayoutV3/PaddleOCR-VL, and valid `sample_page.md`/`sample_page.json` recovered the sample text and table. Archived logs show clean API startup and shutdown, with no `ERROR`/`Traceback`. Evidence is `../gpu_smoke_w8_evidence.zip` (SHA-256 `fd6a88c2082381546329e9c08a59ef7fb561681c496c60b7d8c855e70a4535b9`). This is not an accuracy or throughput benchmark, and it did not download from Hugging Face.
- W4A16 likewise passed **one synthetic-page functional GPU smoke test** in the user-run Kaggle session. The client log verified 14 model files and the executed bundle; the server log identifies `GPTQ_W4A16_G128`, reached API startup and graceful shutdown, and has no `ERROR`/`Traceback`. Its valid JSON has six parsed regions, and Markdown recovered the synthetic page text and table (the same Markdown as W8A16). Evidence is `../gpu_smoke_w4_evidence.zip` (SHA-256 `d72f762ba1f808be20285eb4b28a63cf283d795688e26eaff45b73b840299b26`). This is not an accuracy or throughput benchmark and was not downloaded from Hugging Face.
- At the original upload checkpoint, the **private** W8A16 repository was at commit `2f94efc` and W4A16 at `2f304d9`, each with 16 curated files. Their weight and tokenizer sizes/hashes matched the local checkpoint manifest. This was remote-file verification only; later model-card commits and clean laptop download/inference are recorded below.

## Laptop validation update (October 30, 2026)

- Fresh private Hugging Face snapshots for W8A16 and W4A16 were downloaded to the laptop. Each of the 14 contracted files (including weights and executable model code) passed `scripts/verify_checkpoints.py` against `config/checkpoint_hashes.json`. The updated model-card commits at this October 30 checkpoint were W8A16 `7ef1ca9dc23790d5f946aeac956dd8b9410c474b` and W4A16 `4cf49155a2fa6985799a4a2cb0b5aeb8c709b250`; both model repositories were **private at that checkpoint**.
- BF16 and both clean-Hub GPTQ snapshots completed real, sequential RTX 4070 Laptop GPU inference with CPU layout on a labeled synthetic page. The localhost browser run, per-model Markdown/JSON, timings and peak device memory are in `laptop_smoke_20261030/` and `presentation_demo_20261030/`. Both low-bit server logs selected Marlin. The 48-second captioned presentation video shortens unchanged waits; it is not a benchmark or proof of arbitrary-page accuracy.
- A one-page PDF completed all three variants and a JPEG completed W4A16 locally. The upload validator rejects invalid, encrypted and multi-page PDFs before GPU work; unsupported types are rejected with a clean replacement path.

## Still pending / excluded

- An examiner-selected page and arbitrary document layouts are not verified until run live. A full re-execution of the three reference notebooks on a separate machine/account was not performed.
- Redistribution rights for third-party benchmark/calibration images and offline wheelhouses have not been established; none are included in this GitHub repository.
- At the October 30 checkpoint, the original code license and public model approval were pending. The owner subsequently authorized an Apache-2.0 code license and public GPTQ model release; see the dated release update below.

## Public release update (October 31, 2026)

- The GitHub code/evidence repository now includes an Apache-2.0 `LICENSE` (SHA-256 `b8c4d7deccd236af023af1c88c4d4e8f0fc2f41914e0fb23a3ec9678fb5a8456`, byte-identical to the upstream model's license text) and a project `NOTICE`. GitHub's public API detected SPDX `Apache-2.0`.
- Both GPTQ Hugging Face repositories are **public and ungated**. Anonymous Hub readback returned W8A16 revision `38a47cf53fc9306da32c653cf8022622be003199` and W4A16 `5e8adfa87f9f652bbae91d06e13f28a2d8e1c8e0`, each with 18 curated files, model-card `license: apache-2.0`, the unchanged upstream LICENSE, and a new modification `NOTICE`. Remote weight LFS sizes and SHA-256 still match the original run manifest. See `public_model_release_20261031.json` for the full file inventory and independent anonymous small-file hash checks.
- Neither public release includes calibration/benchmark images, PP-DocLayoutV3 weights, or offline wheelhouses. The local laptop demo remains a functional smoke test on supplied synthetic and one-page PDF/JPEG inputs, not a benchmark replication or an all-input guarantee.
