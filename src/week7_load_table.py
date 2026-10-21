"""
week7_load_table.py - turns the Week 7 load-test files into a table, a figure and an answer check.

    python src\\week7_load_table.py

Reads (never changes):  results\\load_<q4km|q5km>_c<1|2|4>.json/.jsonl   (Week 7 load test)
                        results\\bench_<q4km|q5km>.jsonl                  (Week 6, one at a time, -np 1)
                        results\\preds_eval_ft_<...>.jsonl                (Week 4 answers)
                        data\\eval_frozen.jsonl                            (the correct answers)
Writes:                 results\\week7_load.md   (table + answer check + notes, for the report)
                        results\\week7_load.csv  (one row per run)
                        results\\week7_load.png  (throughput and time per answer vs requests at once)

Accuracy here is on the same 164 lines only (every second eval line), so it is NOT comparable with the
329-line Week 4 numbers - it only shows whether the answers changed under load.
"""
import csv
import json
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from bench import LEVELS, pct
from score import CAT_FIELDS, metrics, parse, score_one

R = C.RESULTS
RUNS = [("q4km", 1), ("q4km", 2), ("q4km", 4), ("q5km", 1), ("q5km", 2), ("q5km", 4)]
NAMES = {"q4km": "Q4_K_M", "q5km": "Q5_K_M"}
COLOURS = {"q4km": "#2a78d6", "q5km": "#eb6834"}
MARKERS = {"q4km": "o", "q5km": "s"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def load_json(name):
    p = R / name
    if not p.exists():
        sys.exit(f"STOPPED: results\\{name} is missing.")
    return json.loads(p.read_text(encoding="utf-8"))


def eval_rows(name):
    return [r for r in C.read_jsonl(R / name) if r["phase"] == "eval"]


def short_metrics(gold, raws):
    m = metrics([score_one(gold[i], raw) for i, raw in raws.items()])
    return {"action_acc": round(100 * m["recommended_action_acc"], 1),
            "action_macro_f1": round(100 * m["action_macro_f1"], 1),
            "severity_acc": round(100 * m["severity_acc"], 1),
            "service_acc": round(100 * m["service_acc"], 1),
            "id_f1": round(100 * m["id_f1"], 1)}


def main():
    C.verify_frozen()
    gold = {r["id"]: r for r in C.read_jsonl(C.EVAL_FROZEN)}
    rows, changed, problems = [], [], []
    week6 = {}
    for lv, n in RUNS:
        sm = load_json(f"load_{lv}_c{n}.json")
        if sm.get("suffix"):
            sys.exit(f"STOPPED: load_{lv}_c{n}.json is a test run.")
        ev = eval_rows(f"load_{lv}_c{n}.jsonl")
        ids = [r["id"] for r in ev]
        saved = {r["id"]: r["raw_output"] for r in C.read_jsonl(R / LEVELS[lv][2])}
        if lv not in week6:                              # Week 6, same lines, one at a time, -np 1
            w6 = {r["id"]: r for r in eval_rows(f"bench_{lv}.jsonl")}
            missing = [i for i in ids if i not in w6]
            if missing:
                sys.exit(f"STOPPED: {len(missing)} load-test lines are missing from bench_{lv}.jsonl")
            sel = [w6[i] for i in ids]
            lat = [r["latency_ms"] for r in sel]
            week6[lv] = {"rows": w6, "lines_per_min": round(60000 / statistics.fmean(lat), 2),
                         "latency_p50_s": round(pct(lat, 50) / 1000, 2),
                         "latency_p95_s": round(pct(lat, 95) / 1000, 2),
                         "ttft_p50_s": round(pct([r["ttft_ms"] for r in sel], 50) / 1000, 2),
                         "ttft_p95_s": round(pct([r["ttft_ms"] for r in sel], 95) / 1000, 2),
                         "decode_tok_s": round(statistics.median(r["decode_tok_s"] for r in sel), 2)}
        before = short_metrics(gold, {i: saved[i] for i in ids})
        after = short_metrics(gold, {r["id"]: r["raw_output"] for r in ev})
        t, p, m = sm["throughput"], sm["per_request"], sm["memory"]
        rows.append({
            "level": NAMES[lv], "at_once": n, "lines": len(ev),
            "lines_per_min": t["lines_per_minute"], "output_tok_s_total": t["output_tokens_per_s"],
            "latency_p50_s": round(p["latency_ms_p50"] / 1000, 2), "latency_p95_s": round(p["latency_ms_p95"] / 1000, 2),
            "ttft_p50_s": round(p["ttft_ms_p50"] / 1000, 2), "ttft_p95_s": round(p["ttft_ms_p95"] / 1000, 2),
            "decode_tok_s_per_request": p["decode_tok_s_median"],
            "ram_peak_gb": round(m["runner_peak_mb"] / 1000, 2) if m["runner_peak_mb"] else None,
            "same_answer_as_week4": f"{sm['consistency']['same_raw_as_week4']}/{len(ev)}",
            **{f"week4_{k}": v for k, v in before.items()},
            **{f"now_{k}": v for k, v in after.items()},
        })
        for r in ev:
            if r["raw_output"] == saved.get(r["id"]):
                continue
            g = gold[r["id"]]
            _, o_new = parse(r["raw_output"])
            _, o_old = parse(saved.get(r["id"], ""))
            o_new, o_old = o_new or {}, o_old or {}
            fields = [k for k in C.OUTPUT_KEYS if o_new.get(k) != o_old.get(k)]
            s_old, s_new = score_one(g, saved[r["id"]]), score_one(g, r["raw_output"])
            verdict = []
            for f in CAT_FIELDS:
                if s_old[f + "_ok"] != s_new[f + "_ok"]:
                    verdict.append(f"{f} {'now RIGHT' if s_new[f + '_ok'] else 'now WRONG'}")
            if s_old["id_tp"] != s_new["id_tp"] or s_old["id_np"] != s_new["id_np"]:
                verdict.append(f"identifiers {s_old['id_tp']}/{s_old['id_np']} -> {s_new['id_tp']}/{s_new['id_np']} correct")
            w6r = week6[lv]["rows"][r["id"]]
            changed.append({"level": NAMES[lv], "at_once": n, "id": r["id"], "fields": fields,
                            "action": f"{o_old.get('recommended_action')} -> {o_new.get('recommended_action')} "
                                      f"(correct: {g['recommended_action']})",
                            "effect": "; ".join(verdict) or "no field became right or wrong",
                            "reused_now": r["cached_tokens"], "reused_week6": w6r["cached_tokens"]})
        if sm["ollama_version"]["before"] != sm["ollama_version"]["after"]:
            problems.append(f"{NAMES[lv]} x{n}: Ollama version changed during the run")
        for w in sm["warnings"]:
            if "differ from Week 4" not in w:
                problems.append(f"{NAMES[lv]} x{n}: {w}")

    with open(R / "week7_load.csv", "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    # ---- figure: two panels, one measure each, x = requests at once
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    panels = [("lines_per_min", "Eval lines per minute (all requests together)", "Throughput"),
              ("latency_p50_s", "Time per answer, median (s)", "Time each request waits for its answer")]
    for ax, (key, ylabel, title) in zip(axes, panels):
        for lv in ("q4km", "q5km"):
            pts = [r for r in rows if r["level"] == NAMES[lv]]
            xs, ys = [r["at_once"] for r in pts], [r[key] for r in pts]
            ax.plot(xs, ys, color=COLOURS[lv], linewidth=2, zorder=2)
            ax.scatter(xs, ys, s=48, color=COLOURS[lv], marker=MARKERS[lv], edgecolors="white",
                       linewidths=1.5, zorder=3, label=NAMES[lv])
            ax.annotate(NAMES[lv], (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=8.5, color=INK)
            for x, y in zip(xs, ys):
                other = [r[key] for r in rows if r["at_once"] == x and r["level"] != NAMES[lv]]
                above = not other or y >= other[0]          # higher point: value above; lower point: below
                ax.annotate(f"{y:g}", (x, y), xytext=(0, 7 if above else -8), textcoords="offset points",
                            ha="center", va="bottom" if above else "top", fontsize=7.5, color=INK2)
        ax.set_xticks([1, 2, 4])
        ax.set_xlim(0.6, 5.0)
        ax.set_ylim(0, max(r[key] for r in rows) * 1.18)
        ax.set_xlabel("Requests sent at the same time")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=8.5)
    # caption on two lines: on one line it ran off the right edge of the picture
    fig.text(0.01, 0.045, f"{rows[0]['lines']} eval lines (every second one), Ollama 0.34.2 with OLLAMA_NUM_PARALLEL=4, "
             "Core i5-13500H laptop, CPU only.", fontsize=7.5, color=INK2)
    fig.text(0.01, 0.012, "Each request is sent when the previous one from the same 'user' has finished.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(R / "week7_load.png", dpi=200, facecolor="white")

    # ---- markdown
    lines = ["# Week 7 - load test (several requests at once)", "",
             "| Level | At once | Lines per min | Output tok/s (total) | Whole answer p50 / p95 (s) "
             "| First token p50 / p95 (s) | Writing tok/s per request | Peak RAM (GB) | Answers same as Week 4 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for lv in ("q4km", "q5km"):
        w = week6[lv]
        lines.append(f"| {NAMES[lv]} | 1 (Week 6, -np 1) | {w['lines_per_min']} | - | {w['latency_p50_s']} / "
                     f"{w['latency_p95_s']} | {w['ttft_p50_s']} / {w['ttft_p95_s']} | {w['decode_tok_s']} | - | - |")
        for r in [x for x in rows if x["level"] == NAMES[lv]]:
            lines.append(f"| {r['level']} | {r['at_once']} | {r['lines_per_min']} | {r['output_tok_s_total']} "
                         f"| {r['latency_p50_s']} / {r['latency_p95_s']} | {r['ttft_p50_s']} / {r['ttft_p95_s']} "
                         f"| {r['decode_tok_s_per_request']} | {r['ram_peak_gb']} | {r['same_answer_as_week4']} |")
    lines += ["", f"## Did the answers change? (same {rows[0]['lines']} lines; Week 4 answers vs this run)",
              "| Level | At once | Action acc | Action macro-F1 | Severity acc | Service acc | ID F1 |",
              "|---|---|---|---|---|---|---|"]
    for r in rows:
        cells = [f"{r['week4_' + k]} -> {r['now_' + k]}" for k in
                 ("action_acc", "action_macro_f1", "severity_acc", "service_acc", "id_f1")]
        lines.append(f"| {r['level']} | {r['at_once']} | " + " | ".join(cells) + " |")
    lines += ["", "Changed answers (prompt tokens re-used from the previous request: this run vs Week 6):"]
    for c in changed:
        lines.append(f"- {c['level']} x{c['at_once']} {c['id']}: changed {', '.join(c['fields']) or 'formatting only'}; "
                     f"action {c['action']}; {c['effect']}; re-used {c['reused_now']} vs {c['reused_week6']}")
    lines += ["", "## Conditions",
              f"- {rows[0]['lines']} eval lines per run (every second eval line of up to 2000 characters), the same "
              "lines in every run; exact scoring prompt and settings; streamed; 127.0.0.1.",
              "- Ollama 0.34.2 restarted with OLLAMA_NUM_PARALLEL=4: llama-server -np 4 -c 8192 (4 slots of 2048).",
              "- Closed loop: N 'users' (N = 1, 2, 4); each sends a line, waits for the whole answer, sends the next.",
              "- Each run: fresh load, N warm-up dev lines (not counted), then the measured lines.",
              "- The 'Week 6' rows are the single-stream benchmark (-np 1) on the same lines, for comparison "
              "(its lines per minute = 60 / mean time per answer).",
              "- Accuracy in the answer check is on these lines only - not comparable with the 329-line Week 4 figures."]
    if problems:
        lines += ["", "## Problems found"] + [f"- {p}" for p in problems]
    (R / "week7_load.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nSaved: results\\week7_load.md, results\\week7_load.csv and results\\week7_load.png")
    if problems:
        print("\nPROBLEMS FOUND - tell Claude.")


if __name__ == "__main__":
    main()
