# Localhost demo: BF16 vs GPTQ W8A16 vs GPTQ W4A16 on one page

Browser UI (http://127.0.0.1 only) that takes one page (PNG/JPEG, or a single-page PDF),
then for **BF16 → W8A16 → W4A16, strictly one after another** on one GPU:

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

## Status

| Piece | State |
|---|---|
| UI, upload validation, orchestration, process lifecycle | Unit/integration tested on CPU (42 tests) |
| Real vLLM + PaddleOCR adapter | Wired to the existing scripts, **not GPU-tested** |
| BF16 + W8A16 + W4A16 sequentially on an RTX 4070 Laptop 8 GB | **Unverified**. Settings are starting guesses |
| Layout (PP-DocLayoutV3) on CPU while vLLM uses the GPU | Default; **unverified** |
| Layout + VLM on the same GPU (`layout_device: "gpu:0"`) | Not validated (README_SOURCE says the same) |
| PDF input | Passed straight to PaddleOCR; `parse_one_image` refuses anything but exactly one page prediction. **Unverified** |

## Host setup (nothing is installed by this package)

1. Use the two existing Python 3.12 environments from `config/environment_versions.json`
   (vLLM server env; PaddleOCR client env). The UI itself needs only the standard library.
2. Verify each checkpoint once (the launcher repeats this at every start):
   `python scripts/verify_checkpoints.py --manifest config/checkpoint_hashes.json --variant BF16 --checkpoint /path/BF16`
   (repeat for W8A16 and W4A16).
3. Copy `demo/demo_config.example.json` to e.g. `demo/demo_config.json`. Fill in absolute paths only.
4. Start: `cd demo && python -m capstone_demo.app --config demo_config.json --port 8765`,
   then open http://127.0.0.1:8765/. Ctrl+C, SIGTERM and SIGHUP all stop any running vLLM child.

## Host GPU acceptance checklist (do this before presenting)

- [ ] `/api/status` shows 0 problems.
- [ ] BF16 alone: server reaches ready at `gpu_memory_utilization` 0.70 without OOM
      (check `runs/<id>/logs/BF16.server.log`). If OOM or too little KV cache: lower `max_model_len`
      or raise utilization, staying under the VRAM the display leaves free.
- [ ] All three variants complete with `layout_device: "cpu"`. Record CPU layout time separately.
- [ ] Peak device memory (shown when `nvidia-smi` is present; includes other processes) stays under 8 GB.
- [ ] Markdown for each variant matches the page by eye. Similarity % is a text diff against BF16, not accuracy.
- [ ] Quantized variants use `--linear-backend marlin`, which serve_model.py picks automatically from
      `quantization_config`. Confirm Marlin works on the 4070 (Ada, sm_89) in that vLLM build.
- [ ] Only once these pass should the presentation say that local inference works on this laptop.

Latency semantics: "measured" = wall time of one warm `parse_one_image` call (layout + all VLM
region requests + Markdown/JSON save), c=1, this page, this machine. It is not the paper's L4
throughput and must not be compared with it as a replication.

## Tests

`cd demo && PYTHONPATH=. python -m unittest discover -s tests -v` (stdlib only, no GPU, no network).
