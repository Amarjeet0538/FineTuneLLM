"""
bench.py - Week 6: single-stream benchmark of ONE quantisation level on Ollama 0.34.2.

    python src\\bench.py --level q5km --limit 5 --loads 1 --suffix smoke    (quick test, a few minutes)
    python src\\bench.py --level q3km                                       (a real run)
    python src\\bench.py --level q3km --limit 30 --loads 1 --suffix drift   (drift check, at the very end)

Levels: f16, q8, q5km, q4km, q3km  (the five fine-tuned models scored in Week 4).

What one run does, in order:
  1. Safety checks: Ollama is 0.34.2, the model is there, Docker is closed, the charger is in,
     enough free RAM, the same Windows power mode as earlier real runs, no old result file.
  2. Unloads every model, so only this one is ever in memory.
  3. COLD: loads the model fresh --loads times (default 3) and times the first request each
     time (dev lines; nothing can be reused from an earlier request).
  4. WARM-UP: --warmup dev lines (default 3). Recorded, but not part of the numbers.
  5. WARM, the main numbers: every eval line of up to 2000 characters (328 lines), one at a
     time, with the exact scoring prompt and settings (config.build_prompt, run_ollama.OPTIONS),
     streamed so the time to the first token can be measured. Every answer is compared
     character for character with the saved Week 4 answer.
  6. Reads the peak memory of the llama-server process (the program that holds the model),
     unloads the model and writes the results.

Results:  results\\bench_<level>[_<suffix>].jsonl   one line per request
          results\\bench_<level>[_<suffix>].json    the summary
Units: ms = milliseconds. MB / GB = 1,000,000 / 1,000,000,000 bytes (same as the Week 4 size
table), so memory looks about 5% bigger than in Task Manager, which uses 1,048,576-byte MB.
Percentiles (p50, p95) use linear interpolation, the same method as numpy's default.

This script never changes the prompt, the settings or any earlier result file.
"""
import argparse
import ctypes
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

import psutil

import config as C
from run_ollama import OPTIONS

# 127.0.0.1, NOT localhost: on Windows, "localhost" is tried as IPv6 (::1) first. Ollama only
# listens on 127.0.0.1, and Windows keeps retrying the refused IPv6 connection for about 2 s
# before Python falls back - measured on this PC: 2044-2082 ms per call vs 1-21 ms.
URL = "http://127.0.0.1:11434"
EXPECTED_VERSION = "0.34.2"
MAX_LOG_CHARS = 2000        # same limit as src\serve.py (the API refuses longer lines)
RAM_HEADROOM_GB = 1.5       # free RAM needed on top of the model file size
KEEP_ALIVE = "15m"          # same as the scoring runs
RUNNER_NAMES = {"llama-server"}                      # the program Ollama 0.34.2 runs the model in
BLOCKERS = {"docker desktop", "com.docker.backend", "vmmem", "vmmemwsl"}

# level: (Ollama model name, key in results\week4_quant_info.json, saved Week 4 answers on 0.34.2)
LEVELS = {
    "f16":  ("smollm3-ft-f16",  "F16",                   "preds_eval_ft_f16.jsonl"),
    "q8":   ("smollm3-ft-q8",   "Q8_0 (Week 3, reused)", "preds_eval_ft_q8_v0342.jsonl"),
    "q5km": ("smollm3-ft-q5km", "Q5_K_M",                "preds_eval_ft_q5km.jsonl"),
    "q4km": ("smollm3-ft-q4km", "Q4_K_M",                "preds_eval_ft_q4km.jsonl"),
    "q3km": ("smollm3-ft-q3km", "Q3_K_M",                "preds_eval_ft_q3km.jsonl"),
}

# Windows 11 "Power mode" setting (Settings > System > Power & battery)
POWER_MODES = {
    "961cc777-2547-4f9d-8174-7d86181b8a7a": "Best power efficiency",
    "00000000-0000-0000-0000-000000000000": "Balanced",
    "ded574b5-45a0-4f42-8737-46345c09c238": "Best performance",
}


# ---------------------------------------------------------------- small helpers
def stop(msg):
    sys.exit("STOPPED: " + msg)


def http(path, payload=None, timeout=60):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(URL + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def bare(name):
    """'smollm3-ft-q5km:latest' -> 'smollm3-ft-q5km'"""
    name = (name or "").lower()
    return name[:-len(":latest")] if name.endswith(":latest") else name


def proc_name(p):
    n = (p.info.get("name") or "").lower()
    return n[:-4] if n.endswith(".exe") else n


def find_procs(names):
    return [p for p in psutil.process_iter(["name"]) if proc_name(p) in names]


def the_runner():
    """The one llama-server process, or None if there is none."""
    procs = find_procs(RUNNER_NAMES)
    if len(procs) > 1:
        stop(f"{len(procs)} llama-server processes are running - more than one model is loaded. "
             "Run `ollama ps`, then tell Claude.")
    return procs[0] if procs else None


def mem(p):
    """Memory of a process in MB. peak/private only exist on Windows."""
    try:
        m = p.memory_info()
    except (psutil.Error, AttributeError):
        return {"rss_mb": None, "peak_mb": None, "private_mb": None}

    def mb(b):
        return round(b / 1e6, 1) if isinstance(b, int) else None
    return {"rss_mb": mb(m.rss), "peak_mb": mb(getattr(m, "peak_wset", None)),
            "private_mb": mb(getattr(m, "private", None))}


def ram_free_gb():
    return round(psutil.virtual_memory().available / 1e9, 2)


def loaded():
    return [m.get("name", "") for m in http("/api/ps").get("models", [])]


def unload_all(timeout=120):
    """Unload every model and wait until the llama-server process has really gone."""
    for name in loaded():
        http("/api/generate", {"model": name, "keep_alive": 0})
    t0 = time.time()
    while loaded() or find_procs(RUNNER_NAMES):
        if time.time() - t0 > timeout:
            stop("a model did not unload within 2 minutes. Run `ollama ps` and tell Claude.")
        time.sleep(0.5)


def generate(model, text):
    """One streamed request with the exact scoring prompt and settings.
    Returns (answer text, seconds to first token, seconds in total, Ollama's final message)."""
    payload = {"model": model, "prompt": C.build_prompt(text), "raw": True, "stream": True,
               "keep_alive": KEEP_ALIVE, "options": OPTIONS}
    req = urllib.request.Request(URL + "/api/generate", data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    pieces, first, final = [], None, None
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            for line in resp:                      # Ollama sends one JSON message per line
                if not line.strip():
                    continue
                msg = json.loads(line)
                if msg.get("error"):
                    stop("Ollama error: " + str(msg["error"]))
                if msg.get("response"):
                    if first is None:
                        first = time.perf_counter() - t0
                    pieces.append(msg["response"])
                if msg.get("done"):
                    final = msg
                    break
    except urllib.error.HTTPError as e:
        stop(f"Ollama error {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")
    except (urllib.error.URLError, ConnectionError) as e:
        stop(f"lost the connection to Ollama ({e}). Is the Ollama app still running?")
    total = time.perf_counter() - t0
    if final is None:
        stop("Ollama stopped in the middle of an answer.")
    return "".join(pieces), first, total, final


def make_row(phase, rec, text, first, total, f, saved, runner):
    def ms(ns):
        return round(ns / 1e6, 1) if isinstance(ns, (int, float)) else None
    prompt_tokens = f.get("prompt_eval_count") or 0
    cached = f.get("prompt_eval_cached_count")          # tokens re-used from the previous request
    new_tokens = prompt_tokens - (cached or 0)
    prompt_ms, decode_ms = ms(f.get("prompt_eval_duration")), ms(f.get("eval_duration"))
    out_tokens = f.get("eval_count") or 0
    return {
        "phase": phase, "id": rec["id"], "eval_kind": rec.get("eval_kind"),
        "input_chars": len(rec["input"]),
        "ttft_ms": round(first * 1000, 1) if first is not None else None,   # measured here
        "latency_ms": round(total * 1000, 1),                               # measured here
        "server_total_ms": ms(f.get("total_duration")), "load_ms": ms(f.get("load_duration")),
        # time spent outside Ollama's own timer (connecting, sending, reading) - should be tiny
        "overhead_ms": (round(total * 1000 - f["total_duration"] / 1e6, 1)
                        if isinstance(f.get("total_duration"), int) else None),
        "prompt_tokens": prompt_tokens, "cached_tokens": cached, "new_prompt_tokens": new_tokens,
        "prompt_ms": prompt_ms, "output_tokens": out_tokens, "decode_ms": decode_ms,
        "prompt_tok_s": round(new_tokens / (prompt_ms / 1000), 2) if prompt_ms and new_tokens > 0 else None,
        "decode_tok_s": round(out_tokens / (decode_ms / 1000), 2) if decode_ms and out_tokens else None,
        "done_reason": f.get("done_reason"),
        "same_as_week4": (text == saved[rec["id"]]) if rec["id"] in saved else None,
        "runner_pid": runner.pid if runner else None, **(mem(runner) if runner else mem(None)),
        "ram_free_gb": ram_free_gb(),
        "raw_output": text,
    }


def pct(xs, p):
    s = sorted(x for x in xs if x is not None)
    if not s:
        return None
    k = (len(s) - 1) * p / 100
    lo = math.floor(k)
    hi = min(lo + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 1)


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 2) if xs else None


# ---------------------------------------------------------------- Windows-only readings
def power_mode():
    if os.name != "nt":
        return "unknown (not Windows)"
    try:
        buf = (ctypes.c_ubyte * 16)()
        if ctypes.windll.powrprof.PowerGetEffectiveOverlayScheme(buf) != 0:
            return "unknown"
        guid = str(uuid.UUID(bytes_le=bytes(buf)))
        return POWER_MODES.get(guid, "other (" + guid + ")")
    except Exception:
        return "unknown"


def keep_awake(on):
    """Stop Windows from going to sleep while this script runs (the screen may still turn off)."""
    if os.name == "nt":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def threads_from_log():
    """llama-server writes 'system_info: n_threads = N' into Ollama's server.log when it starts."""
    log = Path(os.environ.get("LOCALAPPDATA", "")) / "Ollama" / "server.log"
    try:
        size = log.stat().st_size
        with open(log, "rb") as fh:
            fh.seek(max(0, size - 2_000_000))
            tail = fh.read().decode("utf-8", "replace")
        found = re.findall(r"system_info: n_threads = (\d+)", tail)
        return int(found[-1]) if found else None
    except OSError:
        return None


def runner_details(runner):
    """The llama-server command line and the sha256 in the model file name it was given."""
    try:
        args = runner.cmdline()
    except psutil.Error:
        return None, None
    path = args[args.index("--model") + 1] if "--model" in args[:-1] else ""
    m = re.search(r"sha256-([0-9a-f]{64})", path)
    return args, (m.group(1) if m else None)


# ---------------------------------------------------------------- the run
def main():
    ap = argparse.ArgumentParser(description="Week 6 single-stream benchmark of one level.")
    ap.add_argument("--level", required=True, choices=list(LEVELS))
    ap.add_argument("--limit", type=int, default=0, help="only the first N eval lines (smoke/drift)")
    ap.add_argument("--loads", type=int, default=3, help="fresh loads to time (default 3)")
    ap.add_argument("--warmup", type=int, default=3, help="warm-up dev lines, not counted (default 3)")
    ap.add_argument("--suffix", default="", help="added to the file names, e.g. smoke or drift")
    ap.add_argument("--overwrite", action="store_true", help="redo a run whose files already exist")
    args = ap.parse_args()

    if args.limit and not args.suffix:
        stop("--limit needs --suffix (for example smoke or drift), so a short run can never "
             "be mistaken for a full one.")
    if args.loads < 1 or args.warmup < 0 or args.limit < 0:
        stop("--loads must be 1 or more; --warmup and --limit can't be negative.")
    if args.suffix and not re.fullmatch(r"[a-z0-9]+", args.suffix):
        stop("--suffix must be lower-case letters/numbers only, e.g. smoke")

    model, info_key, preds_file = LEVELS[args.level]
    name = f"bench_{args.level}" + (f"_{args.suffix}" if args.suffix else "")
    out_rows, out_sum = C.RESULTS / (name + ".jsonl"), C.RESULTS / (name + ".json")
    if (out_rows.exists() or out_sum.exists()) and not args.overwrite:
        stop(f"results\\{out_rows.name} already exists. Add --overwrite only if you want to redo this run.")

    # ---- the data
    C.verify_frozen()
    all_eval = C.read_jsonl(C.EVAL_FROZEN)
    evals = [r for r in all_eval if len(r["input"]) <= MAX_LOG_CHARS]
    too_long = [r["id"] for r in all_eval if len(r["input"]) > MAX_LOG_CHARS]
    if args.limit:
        evals = evals[:args.limit]
    dev = [r for r in C.read_jsonl(C.DEV) if len(r["input"]) <= MAX_LOG_CHARS]
    cold_lines = dev[:args.loads]
    warm_lines = dev[args.loads:args.loads + args.warmup]
    saved = {r["id"]: r["raw_output"] for r in C.read_jsonl(C.RESULTS / preds_file)}
    quant_info = json.loads((C.RESULTS / "week4_quant_info.json").read_text(encoding="utf-8"))
    week4_sha = quant_info["files"][info_key]["sha256"]

    # ---- safety checks
    try:
        version = http("/api/version", timeout=10)["version"]
    except (urllib.error.URLError, ConnectionError, OSError):
        stop("can't reach Ollama at " + URL + ". Is the Ollama app running?")
    if version != EXPECTED_VERSION:
        stop(f"Ollama is {version}, not {EXPECTED_VERSION}. Don't run anything else - tell Claude.")
    tags = {bare(m.get("name")): m for m in http("/api/tags").get("models", [])}
    if model not in tags:
        stop(f"model {model} is not in Ollama. Models found: {sorted(tags)}")
    file_gb = tags[model].get("size", 0) / 1e9
    blockers = sorted({p.info.get("name") for p in find_procs(BLOCKERS)})
    if blockers:
        stop("quit Docker Desktop first (tray icon > Quit). Still running: " + ", ".join(blockers))
    battery = psutil.sensors_battery()
    if battery is not None and battery.power_plugged is False:
        stop("the laptop is on battery. Plug in the charger.")
    on_charger = None if battery is None else battery.power_plugged
    mode = power_mode()
    earlier = {}
    for f in C.RESULTS.glob("bench_*.json"):
        if f == out_sum or f.stem.endswith("_smoke"):
            continue
        try:
            earlier[f.name] = json.loads(f.read_text(encoding="utf-8"))["machine"]["power_mode"]
        except (OSError, ValueError, KeyError):
            pass
    different = {k: v for k, v in earlier.items() if v != mode}
    if different:
        stop(f"Windows power mode is now '{mode}', but earlier runs used {different}. "
             "Switch it back before running.")

    unload_all()
    free_start = ram_free_gb()
    need = round(file_gb + RAM_HEADROOM_GB, 1)
    if free_start < need:
        stop(f"only {free_start} GB of RAM is free and this level needs about {need} GB. "
             "Close other apps (browser, Antigravity, ...) and try again.")
    warnings = []
    if any("antigravity" in proc_name(p) for p in psutil.process_iter(["name"])):
        warnings.append("Antigravity was open during this run (it uses RAM and CPU).")

    print(f"Week 6 benchmark | level {args.level} = {model} ({file_gb:.2f} GB file) | Ollama {version}")
    print(f"  power mode: {mode} | charger: {on_charger} | free RAM: {free_start} GB")
    print(f"  {len(cold_lines)} cold loads, {len(warm_lines)} warm-up lines, {len(evals)} eval lines "
          f"({len(too_long)} eval line(s) over {MAX_LOG_CHARS} characters skipped)")
    print("  Don't use the laptop for anything heavy until it says DONE.\n")

    keep_awake(True)
    started = datetime.now().astimezone()
    rows = []
    try:
        with open(out_rows, "w", encoding="utf-8", newline="\n") as fout:
            def record(r):
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                fout.flush()
                rows.append(r)

            # ---- COLD: a fresh load before each of these requests
            for i, rec in enumerate(cold_lines, 1):
                unload_all()
                text, first, total, f = generate(model, rec["input"])
                runner = the_runner()
                if runner is None:
                    stop("could not find the llama-server process after loading the model.")
                r = make_row("cold", rec, text, first, total, f, saved, runner)
                record(r)
                print(f"  cold load {i}/{len(cold_lines)}: loading {r['load_ms']:.0f} ms, "
                      f"first token after {r['ttft_ms']} ms, whole answer {r['latency_ms']:.0f} ms")
            runner_args, blob_sha = runner_details(runner)
            threads = threads_from_log()

            # ---- WARM-UP: not counted
            for rec in warm_lines:
                text, first, total, f = generate(model, rec["input"])
                record(make_row("warmup", rec, text, first, total, f, saved, the_runner()))

            # ---- WARM: the main numbers
            runner = the_runner()
            if runner is None:
                stop("the model is no longer loaded after the warm-up.")
            t_start = time.time()
            for i, rec in enumerate(evals, 1):
                text, first, total, f = generate(model, rec["input"])
                if not runner.is_running():       # same process as before the eval lines?
                    stop("the model was reloaded in the middle of the run, so the numbers are mixed. "
                         "Rerun this level with --overwrite.")
                record(make_row("eval", rec, text, first, total, f, saved, runner))
                if i == 1 or i % 10 == 0 or i == len(evals):
                    per = (time.time() - t_start) / i
                    print(f"  {i}/{len(evals)}  {per:.1f} s per line  ~{per * (len(evals) - i) / 60:.0f} min left")

            end_mem = mem(runner)
            ps = [m for m in http("/api/ps").get("models", []) if bare(m.get("name")) == model]
        version_after = http("/api/version", timeout=10)["version"]
        unload_all()
    except KeyboardInterrupt:
        sys.exit(f"\nStopped by you. results\\{out_rows.name} is incomplete - rerun with --overwrite.")
    finally:
        keep_awake(False)
    finished = datetime.now().astimezone()

    # ---- summary
    cold = [r for r in rows if r["phase"] == "cold"]
    ev = [r for r in rows if r["phase"] == "eval"]
    compared = [r for r in ev if r["same_as_week4"] is not None]
    different_ids = [r["id"] for r in compared if not r["same_as_week4"]]
    lat = [r["latency_ms"] for r in ev]
    out_sum_tokens = sum(r["output_tokens"] for r in ev)
    out_sum_secs = sum((r["decode_ms"] or 0) for r in ev) / 1000
    ps_size = ps[0].get("size") if ps else None
    ps_vram = ps[0].get("size_vram") if ps else None

    if version_after != version:
        warnings.append(f"OLLAMA VERSION CHANGED during the run: {version} -> {version_after}. Don't use these numbers.")
    if blob_sha and blob_sha != week4_sha:
        warnings.append("the model file Ollama loaded is NOT the Week 4 file (sha256 differs).")
    if blob_sha is None:
        warnings.append("could not read which model file llama-server loaded.")
    if different_ids:
        warnings.append(f"{len(different_ids)} answer(s) differ from Week 4: {different_ids[:10]}")
    if ps_vram:
        warnings.append(f"Ollama reports {ps_vram / 1e6:.0f} MB of the model on a GPU - not CPU-only.")
    low_ram = min((r["ram_free_gb"] for r in ev), default=None)
    overhead = med([r["overhead_ms"] for r in ev])
    if overhead is not None and overhead > 100:
        warnings.append(f"each request spent ~{overhead:.0f} ms outside Ollama (connection problem?) "
                        "- the measured times are too high.")
    if low_ram is not None and low_ram < 1.0:
        warnings.append(f"free RAM fell to {low_ram} GB during the run - Windows may have used the page file.")

    summary = {
        "level": args.level, "model": model, "suffix": args.suffix,
        "started": started.isoformat(timespec="seconds"), "finished": finished.isoformat(timespec="seconds"),
        "minutes": round((finished - started).total_seconds() / 60, 1),
        "ollama_version": {"before": version, "after": version_after},
        "runner": {"program": "llama-server", "threads_from_log": threads,
                   "model_blob_sha256": blob_sha, "week4_file_sha256": week4_sha,
                   "same_file_as_week4": (blob_sha == week4_sha) if blob_sha else None,
                   "args": runner_args},
        "machine": {"power_mode": mode, "on_charger": on_charger,
                    "cpu_physical_cores": psutil.cpu_count(logical=False),
                    "cpu_logical": psutil.cpu_count(logical=True),
                    "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
                    "ram_free_gb_at_start": free_start},
        "settings": {"options": OPTIONS, "raw": True, "stream": True, "keep_alive": KEEP_ALIVE,
                     "eval_lines": len(ev), "skipped_too_long": too_long,
                     "cold_loads": len(cold), "warmup_lines": len(warm_lines),
                     "percentiles": "linear interpolation (numpy default)",
                     "units": "ms = milliseconds; MB/GB = 10^6/10^9 bytes"},
        "cold": {"n": len(cold),
                 "load_ms": [r["load_ms"] for r in cold], "ttft_ms": [r["ttft_ms"] for r in cold],
                 "latency_ms": [r["latency_ms"] for r in cold],
                 "load_ms_median": med([r["load_ms"] for r in cold]),
                 "ttft_ms_median": med([r["ttft_ms"] for r in cold]),
                 "latency_ms_median": med([r["latency_ms"] for r in cold])},
        "warm": {"n": len(ev),
                 "ttft_ms_p50": pct([r["ttft_ms"] for r in ev], 50),
                 "ttft_ms_p95": pct([r["ttft_ms"] for r in ev], 95),
                 "latency_ms_p50": pct(lat, 50), "latency_ms_p95": pct(lat, 95), "latency_ms_mean": mean(lat),
                 "lines_per_minute": round(60000 / statistics.fmean(lat), 2) if lat else None,
                 "overhead_ms_median": overhead,
                 "decode_tok_s_median": med([r["decode_tok_s"] for r in ev]),
                 "decode_tok_s_overall": round(out_sum_tokens / out_sum_secs, 2) if out_sum_secs else None,
                 "prompt_tok_s_median": med([r["prompt_tok_s"] for r in ev]),
                 "prompt_tokens_mean": mean([r["prompt_tokens"] for r in ev]),
                 "cached_tokens_mean": mean([r["cached_tokens"] for r in ev]),
                 "new_prompt_tokens_mean": mean([r["new_prompt_tokens"] for r in ev]),
                 "output_tokens_mean": mean([r["output_tokens"] for r in ev]),
                 "length_stops": sum(r["done_reason"] == "length" for r in ev)},
        "memory": {"runner_peak_mb": end_mem["peak_mb"], "runner_private_mb_end": end_mem["private_mb"],
                   "runner_rss_mb_first_eval": ev[0]["rss_mb"] if ev else None,
                   "runner_rss_mb_last_eval": ev[-1]["rss_mb"] if ev else None,
                   "ollama_ps_size_mb": round(ps_size / 1e6, 1) if ps_size else None,
                   "ollama_ps_size_vram_mb": round(ps_vram / 1e6, 1) if ps_vram is not None else None,
                   "ram_free_gb_lowest": low_ram},
        "consistency": {"same_as_week4": len(compared) - len(different_ids), "compared": len(compared),
                        "different_ids": different_ids},
        "warnings": warnings,
    }
    out_sum.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    w, m = summary["warm"], summary["memory"]
    print(f"\n==== DONE: {args.level} ({model}) in {summary['minutes']} min ====")
    print(f"Answers identical to Week 4: {summary['consistency']['same_as_week4']}/{len(compared)}")
    print(f"WARM ({w['n']} lines)  first token: p50 {w['ttft_ms_p50']} ms, p95 {w['ttft_ms_p95']} ms")
    print(f"                whole answer: p50 {w['latency_ms_p50']} ms, p95 {w['latency_ms_p95']} ms "
          f"(~{w['lines_per_minute']} lines per minute)")
    print(f"                writing speed: {w['decode_tok_s_median']} tokens/s (median), "
          f"{w['decode_tok_s_overall']} overall")
    print(f"                prompt reading: {w['prompt_tok_s_median']} tokens/s on ~{w['new_prompt_tokens_mean']} "
          f"new tokens per line ({w['cached_tokens_mean']} re-used)")
    print(f"                time outside Ollama: {w['overhead_ms_median']} ms per line (should be tiny)")
    print(f"COLD ({summary['cold']['n']} fresh loads, medians)  loading {summary['cold']['load_ms_median']} ms, "
          f"whole first answer {summary['cold']['latency_ms_median']} ms")
    print(f"MEMORY  llama-server peak {m['runner_peak_mb']} MB | on GPU: {m['ollama_ps_size_vram_mb']} MB "
          f"| lowest free RAM {m['ram_free_gb_lowest']} GB")
    print(f"Ollama {version} -> {version_after} | same model file as Week 4: "
          f"{summary['runner']['same_file_as_week4']} | threads {threads} | power mode {mode}")
    for line in warnings:
        print("WARNING: " + line)
    print(f"Saved: results\\{out_sum.name} and results\\{out_rows.name}")


if __name__ == "__main__":
    main()
