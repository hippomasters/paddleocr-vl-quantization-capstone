# Localhost demo: BF16 vs GPTQ W8A16 vs GPTQ W4A16 on one page

Browser UI (http://127.0.0.1 only) that takes one page (PNG/JPEG, or a single-page PDF),
then for the explicitly selected subset of **BF16 → W8A16 → W4A16, strictly one after another** on one GPU:

1. starts `scripts/serve_model.py` (which re-hashes **all** manifest files, including
   the custom `*.py` model code, before `--trust-remote-code`; unlisted executables are refused),
2. waits until `http://127.0.0.1:<port>/v1/models` lists that variant's served name
   (refuses to start if the port is already occupied, so it can't measure a stranger's server),
3. runs `capstone_demo/timed_parse.py` in the Paddle env. It reuses `parse_page.py`'s
   `build_pipeline_kwargs` / `parse_one_image` / `PREDICT_KWARGS` and the same
   `paddle.seed(2026)` + `PaddleOCRVL(...)` construction, adding only timers
   (one warm-up pass, then one measured pass; c=1),
4. stops the server process group so the next variant gets the whole GPU.

Each run is stored in `demo/runs/<run_id>/`: the input page, `result.json`, each variant's real
Markdown/JSON output, `<variant>.timing.json`, and server/client logs. Failures are shown with
log tails. **No output is ever synthesized**; test doubles exist only under `demo/tests/`.
Select one, two, or all three models in the browser. Unselected models appear as skipped; the
run still uses the fixed order for selected models. BF16-relative speedup and similarity are
shown only when BF16 has a successful result. The UI displays full Markdown and structured
JSON as plain text, never rendered HTML, and offers attachment downloads for completed models.
The POST `/api/runs` API takes the image bytes as before, plus the required
`X-Capstone-Variants: BF16,W8A16,W4A16` header (any nonempty distinct subset is accepted).
Downloads use only `/api/runs/<run_id>/outputs/<variant>/markdown|json`; arbitrary paths,
skipped or incomplete outputs, and symlinks are not served.

## Status

| Piece | State |
|---|---|
| UI, upload validation, orchestration, process lifecycle | 52 CPU tests pass; the real localhost browser upload and sequential run completed |
| Real vLLM + PaddleOCR adapter | All three completed on a synthetic smoke page on the RTX 4070 Laptop 8 GB GPU |
| BF16 → W8A16 → W4A16 sequentially | Verified for that page at 0.72 GPU-memory utilization, 8192 max model length and one sequence; see `../evidence/laptop_smoke_20261030/` |
| Layout (PP-DocLayoutV3) on CPU while vLLM uses the GPU | Verified for that page; layout time is included in page-parse wall time, not isolated |
| Layout + VLM on the same GPU (`layout_device: "gpu:0"`) | Not validated; CPU layout is the tested configuration |
| PDF input | A one-page PDF completed all three models on the laptop. Poppler `pdfinfo` now rejects encrypted, invalid, or multi-page PDFs before starting GPU work; users must export the desired page first. Other PDFs remain unverified. |

The synthetic page is **not** a benchmark, accuracy test, examiner-selected page, or proof that
all image/PDF layouts will work. The 7-minute browser recording in the local evidence folder
shows the real laptop run; the actual Markdown/JSON and timings are saved separately.

## Host setup (nothing is installed by this package)

1. Use separate Python 3.12 vLLM and PaddleOCR environments. The original study's
   `config/environment_versions.json` is **not** the laptop environment; actual installed
   versions and settings are in the local evidence folder's `laptop_environment.json`.
   PaddleX needs its `genai-client` extra for the vLLM-server client path. The UI uses
   the standard library, plus the `pdfinfo` executable from Poppler for PDF preflight.
2. Verify each checkpoint once (the launcher repeats this at every start):
   `python scripts/verify_checkpoints.py --manifest config/checkpoint_hashes.json --variant BF16 --checkpoint /path/BF16`
   (repeat for W8A16 and W4A16).
3. Copy `demo/demo_config.example.json` to e.g. `demo/demo_config.json`. Fill in absolute paths only.
4. Start: `cd demo && python -m capstone_demo.app --config demo_config.json --port 8765`,
   then open http://127.0.0.1:8765/. Ctrl+C, SIGTERM and SIGHUP all stop any running vLLM child.

## Host GPU acceptance checklist (do this before presenting)

- [x] `/api/status` showed 0 problems and browser upload started the run.
- [x] All three variants completed on the synthetic page with `layout_device: "cpu"`,
      `gpu_memory_utilization: 0.72`, `max_model_len: 8192`, one sequence, and warmup enabled.
- [x] Peak device memory was 5596/5602/5614 MiB (BF16/W8/W4) of 8188 MiB.
- [x] Outputs recovered the synthetic page text and table; similarity to BF16 was 100%
      **for this one synthetic page only**, not an accuracy score.
- [x] Both quantized servers logged `MarlinLinearKernel` on the 4070 (Ada, sm_89).
- [x] A one-page PDF completed all three models, and a JPEG completed W4A16 on the laptop.
- [ ] Validate the actual examiner-selected page live; unsupported formats, multi-page PDFs,
      oversized/corrupt uploads and arbitrary document layouts are not guaranteed.

Latency semantics: "measured" = wall time of one warm `parse_one_image` call (layout + all VLM
region requests + Markdown/JSON save), c=1, this page, this machine. It is not the paper's L4
throughput and must not be compared with it as a replication.

## Tests

`cd demo && PYTHONPATH=. python -m unittest discover -s tests -v` (52 CPU tests; no GPU or network).
