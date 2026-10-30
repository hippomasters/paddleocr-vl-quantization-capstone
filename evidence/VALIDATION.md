# Local release validation

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
- Private Hugging Face repository `adityatrha/PaddleOCR-VL-1.6-GPTQ-W8A16-G128` has commit `2f94efc` with all 16 curated files. The Hub tree reports `model.safetensors` at 1,525,406,352 bytes and SHA-256 `b9f9464a0b2dddfdfa1d608b350e28eedb5e0aca78777c45abede9d850baee08`, matching the archived checkpoint; `tokenizer.json` likewise matches its recorded size and SHA-256. Repository visibility was read back as **private**. This is remote file verification, not a clean snapshot download/inference test. The W4 private repository `adityatrha/PaddleOCR-VL-1.6-GPTQ-W4A16-G128` now has commit `2f304d9` and all 16 curated files. Its 14 contracted assets match their archived byte counts, with the weight and tokenizer Xet SHA-256 matching the local manifest; the repo was read back as **private**. Neither variant has yet had a clean Hub snapshot download and GPU inference test.

## Not yet verified

- A **clean download from private Hugging Face model repositories** followed by GPU inference for both variants; the W8A16 and W4A16 smoke tests used attached, locally staged candidate checkpoints instead.
- A full re-execution of the three reference notebooks on a separate account/machine. Their source is preserved exactly, but the cleaned notebook copies intentionally contain no outputs.
- Public distribution rights for third-party benchmark images, calibration data, and the offline wheelhouse. These are excluded from the candidate model repos and reproduction package.
- Owner-selected license for the separate reproduction code and permission for public visibility. W8 and W4 are uploaded **privately**. No public publication has occurred.

**Status:** private, reviewed release candidates; not publication-ready until the remaining gates are addressed.
