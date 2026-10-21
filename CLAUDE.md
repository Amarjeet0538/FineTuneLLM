# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research project, not a product. SmolLM3-3B is fine-tuned (QLoRA) to turn one raw log line (LogHub-2k: HDFS, BGL, OpenStack, Spark) into a 4-key JSON record (`service`, `severity`, `identifiers`, `recommended_action`). The aim is to measure how much accuracy survives quantisation (F16, Q8_0, Q5_K_M, Q4_K_M, Q3_K_M) against memory, latency and throughput. All reported numbers are in `README.md` and `results/`. The project is finished; only the written report is left.

There are no tests, linter or build step. Scripts are plain Python run from the repo root. Docstrings show Windows paths (`python src\\x.py`); forward slashes work too. Each script's module docstring is its usage doc, so read it before running the script.

## Commands

```
pip install -r requirements.txt            # full pipeline (pandas, groq, google-genai, psutil, matplotlib)
pip install -r requirements-serve.txt      # API only (fastapi, uvicorn, httpx)

ollama create smollm3-ft-q5km -f models/Modelfile.q5km   # GGUFs come from HF, not git (*.gguf is ignored)

python src/run_ollama.py --model smollm3-ft-q5km --tag ft_q5km [--split dev] [--limit 5]
python src/score.py results/preds_eval_ft_q5km.jsonl [--split dev]

uvicorn serve:app --app-dir src --port 8000     # API + page at :8000, /docs, /health, POST /triage
docker compose up -d --build                    # Ollama 0.34.2 + API on :8080 (demo only)

python src/bench.py --level q5km [--limit 5 --loads 1 --suffix smoke]
python src/loadtest.py --level q4km --concurrency 2   # needs Ollama started with OLLAMA_NUM_PARALLEL=4
python src/week6_table.py && python src/week7_figure.py && python src/week7_load_table.py
```

`bench.py` and `loadtest.py` only run on Windows (`ctypes.windll`). They refuse to start unless Ollama is exactly 0.34.2, Docker is closed, the charger is connected, the power mode matches, there is enough free RAM and no result file exists yet.

## Pipeline (data flows in this order)

1. `build_dataset.py`: LogHub structured CSVs in `data/raw/` become `data/labeled/records_partial.jsonl`. `service`, `severity` and `identifiers` are ground truth: identifiers come from matching `Content` against LogHub's corrected `EventTemplate`. Drain3 is deliberately not used.
2. `label_with_teacher.py`: `recommended_action` is the only field labelled by the teacher model (groq/gemini/mock). There is one call per (event_type, severity) group, cached in `action_cache.jsonl` and copied to every line in the group. `cleanup_bgl_dumps.py` is a one-off fix on top of that.
3. `split_data.py`, then a human reviews `eval_review.csv`, then `freeze_eval.py` (run once). The split is by template, so unseen templates never reach training. Freezing writes `data/eval_frozen.jsonl` and its sha256. Corrections made in review apply to train and dev as well.
4. `build_train_set.py` writes `data/week3/*_sft.jsonl`: dedupes lines and caps `no_action` at 20 per group.
5. Kaggle notebooks (`notebooks/`) do the fine-tune, the merge plus GGUF quantisation, and the HF upload.
6. `run_ollama.py` writes `results/preds_*.jsonl`, then `score.py` writes `results/scores_*.json`.
7. `bench.py`/`loadtest.py` write `results/bench_*`/`load_*`, and the `week*_table.py`/`week7_figure.py` scripts turn those into tables and figures.

## Invariants that must not be broken

- **`src/config.py` is the single source of truth.** It holds the paths, `SEED=42`, the label vocabulary (`ACTIONS`, `ACTION_MERGE`, `SEVERITIES`, `OUTPUT_KEYS`), the frozen prompt (`build_prompt`, a hand-written SmolLM3 no-think chat template) and `format_target`. Training, scoring, benchmarking and serving all import from it. Changing the prompt or the vocabulary invalidates every result.
- **Scoring, benchmarking and serving use the same prompt, settings and parser.** These are `config.build_prompt`, `run_ollama.OPTIONS` (temperature 0, seed, `num_ctx` 2048, `num_predict` 384, stop `<|im_end|>`, raw mode, no `format`/JSON forcing) and `score.parse`. `serve.py` and `bench.py` import these rather than copying them, so the API answers exactly as the scored model did.
- **The eval set is frozen.** `config.verify_frozen()` stops the script if `eval_frozen.jsonl` doesn't match `eval_frozen.sha256`. `split_data.py` and `freeze_eval.py` refuse to run once it exists. `.gitattributes` marks `*.jsonl -text` so line endings never change the hash.
- Scripts never overwrite earlier result files. Pass `--suffix`/`--tag` for new runs.
- Ollama model names follow `smollm3-ft-<level>` (see `bench.LEVELS`). `serve.py` defaults to `smollm3-ft-q5km` and reads the `OLLAMA_URL`/`MODEL` env vars. It uses `127.0.0.1` rather than `localhost` on purpose: on Windows, `localhost` tries IPv6 first and each request loses about 2 s.
- `MAX_LOG_CHARS = 2000` in `serve.py`, and the benchmarks skip the one eval line over that length.
- The Dockerfile copies only `config.py`, `run_ollama.py`, `score.py`, `serve.py` and `ui.html`. If `serve.py` gets a new import, add that file to the Dockerfile.
