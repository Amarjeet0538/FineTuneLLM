"""
week6_table.py - turns the Week 6 benchmark files into one results table.

    python src\\week6_table.py

Reads (never changes):  results\\bench_<level>.json and .jsonl for the 5 levels,
                        results\\bench_q3km_drift.json/.jsonl, results\\week4_quant_info.json
Writes:                 results\\week6_table.md   (table + conditions + notes, for the report)
                        results\\week6_table.csv  (one row per level, for the Week 7 figure)

GB = 1,000,000,000 bytes (same as the Week 4 size table). s = seconds.
"""
import csv
import json
import statistics
import sys

import config as C
from bench import LEVELS

ORDER = ["q3km", "q4km", "q5km", "q8", "f16"]
NAMES = {"q3km": "Q3_K_M", "q4km": "Q4_K_M", "q5km": "Q5_K_M", "q8": "Q8_0", "f16": "F16"}
R = C.RESULTS


def load_json(name):
    p = R / name
    if not p.exists():
        sys.exit(f"STOPPED: results\\{name} is missing.")
    return json.loads(p.read_text(encoding="utf-8"))


def load_rows(name):
    with open(R / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def s(ms):
    return round(ms / 1000, 2) if isinstance(ms, (int, float)) else None


def gb(mb):
    return round(mb / 1000, 2) if isinstance(mb, (int, float)) else None


def main():
    info = load_json("week4_quant_info.json")["files"]
    rows, problems = [], []
    for lv in ORDER:
        sm = load_json(f"bench_{lv}.json")
        if sm.get("suffix"):
            sys.exit(f"STOPPED: bench_{lv}.json is a test run ({sm['suffix']}).")
        w, m, cold, con = sm["warm"], sm["memory"], sm["cold"], sm["consistency"]
        file_gb = round(info[LEVELS[lv][1]]["bytes"] / 1e9, 2)
        loads = cold["load_ms"]
        reload_ms = statistics.median(loads[1:]) if len(loads) > 1 else None
        rows.append({
            "level": NAMES[lv],
            "file_gb": file_gb,
            "ram_after_load_gb": gb(m["runner_rss_mb_first_eval"]),
            "ram_peak_gb": gb(m["runner_peak_mb"]),
            "ttft_p50_s": s(w["ttft_ms_p50"]), "ttft_p95_s": s(w["ttft_ms_p95"]),
            "latency_p50_s": s(w["latency_ms_p50"]), "latency_p95_s": s(w["latency_ms_p95"]),
            "latency_mean_s": s(w["latency_ms_mean"]),
            "lines_per_min": w["lines_per_minute"],
            "decode_tok_s": w["decode_tok_s_median"],
            "prompt_tok_s": w["prompt_tok_s_median"],
            "first_load_s": s(loads[0]) if loads else None,
            "reload_s": s(reload_ms),
            "same_as_week4": f"{con['same_as_week4']}/{con['compared']}",
            "eval_lines": w["n"],
            "minutes": sm["minutes"],
            "started": sm["started"],
        })
        if sm["ollama_version"]["before"] != sm["ollama_version"]["after"]:
            problems.append(f"{NAMES[lv]}: Ollama version changed during the run")
        if con["same_as_week4"] != con["compared"]:
            problems.append(f"{NAMES[lv]}: {con['compared'] - con['same_as_week4']} answers differ from Week 4")
        for wn in sm["warnings"]:
            if lv == "q8" and "NOT the Week 4 file" in wn:
                continue                       # explained in the notes (tensor order only)
            problems.append(f"{NAMES[lv]}: {wn}")

    first = load_json("bench_q3km.json")
    q8 = load_json("bench_q8.json")
    drift_sum = load_json("bench_q3km_drift.json")
    start = [r for r in load_rows("bench_q3km.jsonl") if r["phase"] == "eval"][:drift_sum["warm"]["n"]]
    end = [r for r in load_rows("bench_q3km_drift.jsonl") if r["phase"] == "eval"]
    if [r["id"] for r in start] != [r["id"] for r in end]:
        sys.exit("STOPPED: the drift run did not use the same lines as the Q3_K_M run.")
    drift = {}
    for key in ["decode_tok_s", "prompt_tok_s", "ttft_ms", "latency_ms"]:
        a = statistics.median(r[key] for r in start)
        b = statistics.median(r[key] for r in end)
        drift[key] = round(100 * (b - a) / a, 1)

    # ---- CSV (for the Week 7 figure)
    with open(R / "week6_table.csv", "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    # ---- Markdown (for the report)
    mc = first["machine"]
    run = first["runner"]
    st = first["settings"]
    lines = [
        "# Week 6 - single-stream benchmark results",
        "",
        "| Level | File (GB) | RAM after load (GB) | Peak RAM (GB) | First token p50 / p95 (s) "
        "| Whole answer p50 / p95 (s) | Lines per min | Writing (tok/s) | Prompt reading (tok/s) "
        "| Reload (s) | Answers same as Week 4 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['level']} | {r['file_gb']} | {r['ram_after_load_gb']} | {r['ram_peak_gb']} "
            f"| {r['ttft_p50_s']} / {r['ttft_p95_s']} | {r['latency_p50_s']} / {r['latency_p95_s']} "
            f"| {r['lines_per_min']} | {r['decode_tok_s']} | {r['prompt_tok_s']} "
            f"| {r['reload_s']} | {r['same_as_week4']} |")
    lines += [
        "",
        "## Conditions",
        f"- Runs: {rows[0]['started'][:16]} to {drift_sum['finished'][:16]}, one level at a time, "
        "5-minute pause between levels.",
        f"- Ollama {first['ollama_version']['before']} (checked before and after every run), model run by "
        f"llama-server with {run['threads_from_log']} threads (chosen automatically, not tuned), "
        "CPU only (0 MB on GPU for every level).",
        f"- Laptop: {mc['cpu_physical_cores']} cores / {mc['cpu_logical']} threads, "
        f"{mc['ram_total_gb']} GB RAM, power mode '{mc['power_mode']}', on charger: {mc['on_charger']}. "
        "Docker and other apps closed.",
        f"- Requests: the exact scoring prompt and settings ({json.dumps(st['options'])}, raw mode), "
        "one at a time, streamed.",
        f"- Warm numbers: {rows[0]['eval_lines']} eval lines per level (the {len(st['skipped_too_long'])} line "
        f"over 2000 characters, {', '.join(st['skipped_too_long'])}, is skipped, as in the API), "
        f"after {st['warmup_lines']} warm-up dev lines that are not counted.",
        f"- Ollama re-uses the shared start of the prompt between requests: on average "
        f"{first['warm']['cached_tokens_mean']} of {first['warm']['prompt_tokens_mean']} prompt tokens "
        "were re-used, so only the log-specific part is read each time. This is the default and was also "
        "on during scoring.",
        f"- Reload = median of {st['cold_loads'] - 1} fresh loads with the model file already in Windows' "
        "memory (not from disk).",
        "- Percentiles: linear interpolation (numpy's default). Time outside Ollama (connecting, reading "
        f"the answer): about {first['warm']['overhead_ms_median']} ms per request.",
        "",
        f"## Drift check (the same first {len(start)} Q3_K_M lines, start vs end of the session)",
        f"- Writing speed {drift['decode_tok_s']:+}%, prompt reading {drift['prompt_tok_s']:+}%, "
        f"first token {drift['ttft_ms']:+}%, whole answer {drift['latency_ms']:+}%.",
        "- Differences between levels smaller than these changes are within run-to-run noise.",
        "",
        "## Notes",
        "- The benchmark connects to Ollama at 127.0.0.1, not localhost: on this Windows PC 'localhost' "
        "added about 2.05 s to every request (Windows retries the refused IPv6 connection). Found in the "
        "smoke test and fixed before the real runs.",
        (f"- Q8_0: the file Ollama runs (sha256 {q8['runner']['model_blob_sha256']}) has a different hash "
         f"from the recorded Week 3 file ({q8['runner']['week4_file_sha256']}) only because its tensors are "
         "stored in a different order. Checked 2026-10-15: all 27 settings and all 326 tensors are "
         "byte-identical. Every other level ran exactly the Week 4 file."
         if q8["runner"]["same_file_as_week4"] is False else
         "- Every level ran exactly the file recorded in Week 4 (same sha256)."),
        "- Memory: llama-server copies the whole model into RAM (Ollama turns off memory-mapping when "
        "there is no GPU). RAM grew during each run from 'after load' to 'peak' in identical 43 MB steps "
        "at the same lines for every level, so the growth depends on the prompts, not on the weights. "
        "Its cause was not confirmed.",
        "- Writing speed x file size is about the same for every level (memory bandwidth limit). Q4_K_M "
        "reads prompts fastest because llama.cpp rearranges Q4_K weights into a faster layout on this "
        "CPU and does not do that for Q3_K, Q5_K, Q6_K or Q8_0.",
    ]
    if problems:
        lines += ["", "## Problems found"] + [f"- {p}" for p in problems]
    (R / "week6_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print("\nSaved: results\\week6_table.md and results\\week6_table.csv")
    if problems:
        print("\nPROBLEMS FOUND - tell Claude before using these numbers.")


if __name__ == "__main__":
    main()
