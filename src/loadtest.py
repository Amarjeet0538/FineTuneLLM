"""
loadtest.py - Week 7: what happens when several requests arrive at once (one level, one concurrency).

    python src\\loadtest.py --level q4km --concurrency 2 --limit 8 --suffix smoke   (quick test)
    python src\\loadtest.py --level q4km --concurrency 4                             (a real run)

Needs Ollama started with OLLAMA_NUM_PARALLEL=4 (checked: llama-server must show -np >= concurrency).

What one run does:
  1. Same safety checks as bench.py (Ollama 0.34.2, Docker closed, charger, power mode = Week 6's, RAM).
  2. Unloads every model, loads this one fresh, sends <concurrency> warm-up dev lines at once (not counted).
  3. MEASURED: every second eval line of up to 2000 characters (164 lines, the same lines in every run).
     <concurrency> workers each send one line, wait for the whole answer, then take the next line
     ("closed loop"), with the exact scoring prompt and settings, streamed.
  4. Throughput = measured lines / wall-clock time of the measured part. Per-request times, first token,
     peak RAM of llama-server, answers compared with Week 4 (raw text and the parsed 4-key record).

Results: results\\load_<level>_c<N>[_<suffix>].jsonl (one line per request) and .json (summary).
Reuses bench.py for the request, the checks and the units (ms; MB/GB = 10^6/10^9 bytes).
"""
import argparse
import json
import re
import sys
import threading
import time
from datetime import datetime

import config as C
import bench as B
from score import parse

MAX_RUNNING_MINUTES = 120


def main():
    ap = argparse.ArgumentParser(description="Week 7 load test of one level at one concurrency.")
    ap.add_argument("--level", required=True, choices=["q4km", "q5km"])
    ap.add_argument("--concurrency", required=True, type=int, choices=[1, 2, 4])
    ap.add_argument("--limit", type=int, default=0, help="only the first N of the 164 lines (smoke test)")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.limit and not args.suffix:
        B.stop("--limit needs --suffix (for example smoke).")
    if args.suffix and not re.fullmatch(r"[a-z0-9]+", args.suffix):
        B.stop("--suffix must be lower-case letters/numbers only")

    model, _, preds_file = B.LEVELS[args.level]
    n_conc = args.concurrency
    name = f"load_{args.level}_c{n_conc}" + (f"_{args.suffix}" if args.suffix else "")
    out_rows, out_sum = C.RESULTS / (name + ".jsonl"), C.RESULTS / (name + ".json")
    if (out_rows.exists() or out_sum.exists()) and not args.overwrite:
        B.stop(f"results\\{out_rows.name} already exists. Add --overwrite only if you want to redo this run.")

    # ---- the data: every second eval line (same lines every run)
    C.verify_frozen()
    evals = [r for r in C.read_jsonl(C.EVAL_FROZEN) if len(r["input"]) <= B.MAX_LOG_CHARS][::2]
    if args.limit:
        evals = evals[:args.limit]
    warm = [r for r in C.read_jsonl(C.DEV) if len(r["input"]) <= B.MAX_LOG_CHARS][:n_conc]
    saved = {r["id"]: r["raw_output"] for r in C.read_jsonl(C.RESULTS / preds_file)}

    # ---- safety checks (same as bench.py)
    try:
        version = B.http("/api/version", timeout=10)["version"]
    except OSError:
        B.stop("can't reach Ollama at " + B.URL + ". Is the Ollama app running?")
    if version != B.EXPECTED_VERSION:
        B.stop(f"Ollama is {version}, not {B.EXPECTED_VERSION}. Don't run anything else - tell Claude.")
    tags = {B.bare(m.get("name")): m for m in B.http("/api/tags").get("models", [])}
    if model not in tags:
        B.stop(f"model {model} is not in Ollama.")
    blockers = sorted({p.info.get("name") for p in B.find_procs(B.BLOCKERS)})
    if blockers:
        B.stop("quit Docker Desktop first. Still running: " + ", ".join(blockers))
    battery = B.psutil.sensors_battery()
    if battery is not None and battery.power_plugged is False:
        B.stop("the laptop is on battery. Plug in the charger.")
    mode = B.power_mode()
    try:
        week6_mode = json.loads((C.RESULTS / "bench_q3km.json").read_text(encoding="utf-8"))["machine"]["power_mode"]
    except (OSError, ValueError, KeyError):
        week6_mode = None
    if week6_mode is not None and mode != week6_mode:
        B.stop(f"Windows power mode is '{mode}', but Week 6 used '{week6_mode}'. Switch it back first.")
    B.unload_all()
    free_start = B.ram_free_gb()
    need = round(tags[model].get("size", 0) / 1e9 + 2.0, 1)
    if free_start < need:
        B.stop(f"only {free_start} GB of RAM is free; this run needs about {need} GB. Close other apps.")
    warnings = []
    if any("antigravity" in B.proc_name(p) for p in B.psutil.process_iter(["name"])):
        warnings.append("Antigravity was open during this run (it uses RAM and CPU).")

    print(f"Week 7 load test | {args.level} = {model} | {n_conc} request(s) at once | Ollama {version}")
    print(f"  power mode: {mode} | charger: {None if battery is None else battery.power_plugged} "
          f"| free RAM: {free_start} GB | {len(evals)} measured lines")
    print("  Don't use the laptop for anything heavy until it says DONE.\n")

    lock = threading.Lock()
    rows, errors = [], []
    runner_ref = [None]          # the llama-server process, found once after loading (not per request)

    def run_batch(lines, phase, fout, t_zero):
        """<n_conc> workers; each takes the next line, sends it, waits for the whole answer, repeats."""
        queue = list(lines)
        done = [0]

        def worker(wid):
            while True:
                with lock:
                    if errors or not queue:
                        return
                    rec = queue.pop(0)
                t_send = time.perf_counter() - t_zero
                try:
                    text, first, total, f = B.generate(model, rec["input"])
                except SystemExit as e:                      # bench.generate stops on Ollama errors
                    with lock:
                        errors.append(str(e))
                    return
                row = B.make_row(phase, rec, text, first, total, f, saved, runner_ref[0])
                _, obj = parse(text)
                _, obj_saved = parse(saved.get(rec["id"], ""))
                row.update({"worker": wid, "sent_s": round(t_send, 3),
                            "finished_s": round(time.perf_counter() - t_zero, 3),
                            "same_record_as_week4": (obj == obj_saved) if rec["id"] in saved else None})
                with lock:
                    rows.append(row)
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fout.flush()
                    done[0] += 1
                    if phase == "eval" and (done[0] % 10 == 0 or done[0] == len(lines)):
                        el = time.perf_counter() - t_zero
                        print(f"  {done[0]}/{len(lines)}  {60 * done[0] / el:.1f} lines per minute so far")

        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n_conc)]
        for t in threads:
            t.start()
        deadline = time.time() + MAX_RUNNING_MINUTES * 60
        for t in threads:
            while t.is_alive():
                t.join(timeout=1)
                if time.time() > deadline:
                    errors.append("the run took longer than 2 hours")
                    return
        if errors:
            B.stop(errors[0].replace("STOPPED: ", ""))

    B.keep_awake(True)
    started = datetime.now().astimezone()
    try:
        with open(out_rows, "w", encoding="utf-8", newline="\n") as fout:
            # load + check that Ollama really runs this many requests at once
            t0 = time.perf_counter()
            run_batch(warm[:1], "load", fout, t0)
            runner = runner_ref[0] = B.the_runner()
            if runner is None:
                B.stop("could not find the llama-server process after loading the model.")
            cmd, _ = B.runner_details(runner)
            n_par = int(cmd[cmd.index("-np") + 1]) if cmd and "-np" in cmd[:-1] else 1
            if n_par < n_conc:
                B.stop(f"Ollama runs only {n_par} request(s) at once (-np {n_par}). "
                       "OLLAMA_NUM_PARALLEL was not picked up - tell Claude.")
            run_batch(warm, "warmup", fout, time.perf_counter())
            # the measured part
            t_eval = time.perf_counter()
            run_batch(evals, "eval", fout, t_eval)
            wall_s = time.perf_counter() - t_eval
            if not runner.is_running():
                B.stop("the model was reloaded during the run. Rerun with --overwrite.")
            end_mem = B.mem(runner)
        version_after = B.http("/api/version", timeout=10)["version"]
        B.unload_all()
    except KeyboardInterrupt:
        errors.append("stopped by you")
        sys.exit(f"\nStopped by you. results\\{out_rows.name} is incomplete - rerun with --overwrite.")
    finally:
        B.keep_awake(False)
    finished = datetime.now().astimezone()

    ev = [r for r in rows if r["phase"] == "eval"]
    lat = [r["latency_ms"] for r in ev]
    out_tokens = sum(r["output_tokens"] for r in ev)
    same_raw = sum(1 for r in ev if r["same_as_week4"])
    same_rec = sum(1 for r in ev if r["same_record_as_week4"])
    diff_ids = [r["id"] for r in ev if not r["same_as_week4"]]
    overhead = B.med([r["overhead_ms"] for r in ev])
    if version_after != version:
        warnings.append(f"OLLAMA VERSION CHANGED during the run: {version} -> {version_after}.")
    if overhead is not None and overhead > 100:
        warnings.append(f"each request spent ~{overhead:.0f} ms outside Ollama - times are too high.")
    if diff_ids:
        warnings.append(f"{len(diff_ids)} answer(s) differ from Week 4 in raw text "
                        f"({len(ev) - same_rec} differ in the parsed record): {diff_ids[:10]}")

    summary = {
        "level": args.level, "model": model, "concurrency": n_conc, "suffix": args.suffix,
        "started": started.isoformat(timespec="seconds"), "finished": finished.isoformat(timespec="seconds"),
        "ollama_version": {"before": version, "after": version_after},
        "runner": {"num_parallel": n_par, "args": cmd, "threads_from_log": B.threads_from_log()},
        "machine": {"power_mode": mode, "on_charger": None if battery is None else battery.power_plugged,
                    "ram_free_gb_at_start": free_start},
        "settings": {"options": B.OPTIONS, "lines": len(ev), "line_choice": "every second eval line <= 2000 chars",
                     "warmup_lines": len(warm), "closed_loop_workers": n_conc},
        "throughput": {"wall_s": round(wall_s, 1), "lines_per_minute": round(60 * len(ev) / wall_s, 2),
                       "output_tokens_per_s": round(out_tokens / wall_s, 2)},
        "per_request": {"latency_ms_p50": B.pct(lat, 50), "latency_ms_p95": B.pct(lat, 95),
                        "ttft_ms_p50": B.pct([r["ttft_ms"] for r in ev], 50),
                        "ttft_ms_p95": B.pct([r["ttft_ms"] for r in ev], 95),
                        "decode_tok_s_median": B.med([r["decode_tok_s"] for r in ev]),
                        "overhead_ms_median": overhead},
        "memory": {"runner_peak_mb": end_mem["peak_mb"], "runner_rss_mb_first_eval": ev[0]["rss_mb"] if ev else None,
                   "ram_free_gb_lowest": min((r["ram_free_gb"] for r in ev), default=None)},
        "consistency": {"compared": len(ev), "same_raw_as_week4": same_raw, "same_record_as_week4": same_rec,
                        "different_raw_ids": diff_ids},
        "warnings": warnings,
    }
    out_sum.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    t, p, m = summary["throughput"], summary["per_request"], summary["memory"]
    print(f"\n==== DONE: {args.level}, {n_conc} at once, {len(ev)} lines in {t['wall_s'] / 60:.1f} min ====")
    print(f"THROUGHPUT   {t['lines_per_minute']} lines per minute | {t['output_tokens_per_s']} output tokens/s in total")
    print(f"PER REQUEST  whole answer p50 {p['latency_ms_p50']} ms, p95 {p['latency_ms_p95']} ms | "
          f"first token p50 {p['ttft_ms_p50']} ms | writing {p['decode_tok_s_median']} tok/s per request")
    print(f"MEMORY       llama-server peak {m['runner_peak_mb']} MB | lowest free RAM {m['ram_free_gb_lowest']} GB")
    print(f"ANSWERS      same raw text as Week 4: {same_raw}/{len(ev)} | same parsed record: {same_rec}/{len(ev)}")
    print(f"Ollama {version} -> {version_after} | -np {n_par} | time outside Ollama {overhead} ms")
    for w in warnings:
        print("WARNING: " + w)
    print(f"Saved: results\\{out_sum.name} and results\\{out_rows.name}")


if __name__ == "__main__":
    main()
