"""
week7_figure.py - Week 7: the accuracy-vs-cost figure.

    python src\\week7_figure.py

Reads (never changes):  results\\week6_table.csv               (cost: Week 6 benchmarks)
                        results\\scores_eval_<tag>.json         (accuracy: Week 4 scoring, Ollama 0.34.2)
                        data\\eval_frozen.jsonl                  (for the "always the most common action" line)
Writes:                 results\\week7_tradeoff.png             (the figure, for the report)
                        results\\week7_tradeoff.csv             (the same numbers as a table)

Accuracy = recommended_action on all 329 frozen eval lines. Error bars = the 95% bootstrap intervals
that score.py already computed in Week 4 (1000 resamples of eval lines, fixed seed).
"""
import csv
import json
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")                      # draw straight to a file, no window
import matplotlib.pyplot as plt

import config as C

R = C.RESULTS
# level name in week6_table.csv -> Week 4 score file (all scored on Ollama 0.34.2)
SCORES = {"Q3_K_M": "scores_eval_ft_q3km.json", "Q4_K_M": "scores_eval_ft_q4km.json",
          "Q5_K_M": "scores_eval_ft_q5km.json", "Q8_0": "scores_eval_ft_q8_v0342.json",
          "F16": "scores_eval_ft_f16.json"}
SERIES = [  # (key in the score file, row title, y-axis label, colour, marker)
    ("recommended_action_acc", "Action accuracy", "Action accuracy (%)", "#2a78d6", "o"),
    ("action_macro_f1", "Action macro-F1 (each of the 4 actions counts equally)", "Action macro-F1 (%)",
     "#eb6834", "s"),
]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# Level names sit centred above each error bar, except where they would collide:
BELOW = {"Q3_K_M"}   # Q3_K_M sits on top of Q4_K_M in the time panels, so its name goes under its bar
LEFT = {("recommended_action_acc", "Q4_K_M")}   # in the accuracy row Q4_K_M's name goes left of its bar top


def main():
    path = R / "week6_table.csv"
    if not path.exists():
        sys.exit("STOPPED: results\\week6_table.csv is missing - run src\\week6_table.py first.")
    with open(path, encoding="utf-8") as f:
        cost = {r["level"]: r for r in csv.DictReader(f)}
    C.verify_frozen()
    actions = Counter(r["recommended_action"] for r in C.read_jsonl(C.EVAL_FROZEN))
    top_action, top_n = actions.most_common(1)[0]
    always_top = 100 * top_n / sum(actions.values())

    rows = []
    for level, fname in SCORES.items():
        if level not in cost:
            sys.exit(f"STOPPED: {level} is missing from week6_table.csv")
        sc = json.loads((R / fname).read_text(encoding="utf-8"))["ALL"]
        row = {"level": level,
               "latency_p50_s": float(cost[level]["latency_p50_s"]),
               "latency_p95_s": float(cost[level]["latency_p95_s"]),
               "ram_peak_gb": float(cost[level]["ram_peak_gb"]),
               "file_gb": float(cost[level]["file_gb"]),
               "n_eval": sc["metrics"]["n"]}
        for key, *_ in SERIES:
            lo, hi = sc["ci95"][key]
            row[key] = round(100 * sc["metrics"][key], 1)
            row[key + "_ci_low"] = round(100 * lo, 1)
            row[key + "_ci_high"] = round(100 * hi, 1)
        rows.append(row)

    with open(R / "week7_tradeoff.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- the figure: 2 x 2 panels. Rows = accuracy measure, columns = cost measure.
    # Each panel has ONE measure on ONE axis; the rows have their own y-range.
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), sharex="col", sharey="row")
    cols = [("latency_p50_s", "Time per answer, median (s)"),
            ("ram_peak_gb", "Peak RAM of the model process (GB)")]
    for i, (key, row_title, ylabel, colour, marker) in enumerate(SERIES):
        lo = min(r[key + "_ci_low"] for r in rows)
        hi = max(r[key + "_ci_high"] for r in rows)
        if key == "recommended_action_acc":
            lo = min(lo, always_top)
        for j, (xkey, xlabel) in enumerate(cols):
            ax = axes[i][j]
            if key == "recommended_action_acc":
                ax.axhline(always_top, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
                ax.text(0.15, always_top + 0.25, f"always answering '{top_action}' = {always_top:.1f}%",
                        ha="left", va="bottom", fontsize=7.5, color=INK2)
            xs = [r[xkey] for r in rows]
            ys = [r[key] for r in rows]
            err = [[r[key] - r[key + "_ci_low"] for r in rows], [r[key + "_ci_high"] - r[key] for r in rows]]
            ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor=colour, elinewidth=1.2, alpha=0.4,
                        capsize=0, zorder=2)
            ax.scatter(xs, ys, s=48, color=colour, marker=marker, edgecolors="white", linewidths=1.5,
                       zorder=3)
            for r in rows:
                dx, ha = 0, "center"
                if r["level"] in BELOW:
                    xy, dy, va = (r[xkey], r[key + "_ci_low"]), -3, "top"
                else:
                    xy, dy, va = (r[xkey], r[key + "_ci_high"]), 3, "bottom"
                if (key, r["level"]) in LEFT:
                    dx, dy, ha, va = -6, 0, "right", "center"
                ax.annotate(r["level"], xy, xytext=(dx, dy), textcoords="offset points",
                            ha=ha, va=va, fontsize=8.5, color=INK)
            pad = (hi - lo) * 0.12
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(0, max(xs) * 1.12)
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if i == 1:
                ax.set_xlabel(xlabel)
                ax.tick_params(axis="x", labelbottom=True)
            if j == 0:
                ax.set_ylabel(ylabel)
        axes[i][0].set_title(row_title, loc="left", fontsize=10.5, color=INK, pad=10)
    fig.text(0.01, 0.035, "Fine-tuned SmolLM3-3B at 5 quantisation levels. Accuracy: Week 4 scoring, "
             f"{rows[0]['n_eval']} frozen eval lines, Ollama 0.34.2; bars = 95% bootstrap intervals.",
             fontsize=7.5, color=INK2)
    fig.text(0.01, 0.012, "Cost: Week 6, 328 eval lines, one request at a time, Core i5-13500H laptop, "
             "CPU only, Ollama 0.34.2. GB = 10^9 bytes.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=2.5)
    fig.savefig(R / "week7_tradeoff.png", dpi=200, facecolor="white")

    print("level   time p50 (s)  peak RAM (GB)  action acc [95% CI]   action macro-F1 [95% CI]")
    for r in rows:
        print(f"{r['level']:7s} {r['latency_p50_s']:12.2f}  {r['ram_peak_gb']:13.2f}  "
              f"{r['recommended_action_acc']:5.1f} [{r['recommended_action_acc_ci_low']}, "
              f"{r['recommended_action_acc_ci_high']}]   {r['action_macro_f1']:5.1f} "
              f"[{r['action_macro_f1_ci_low']}, {r['action_macro_f1_ci_high']}]")
    print(f"always '{top_action}': {always_top:.1f}% accuracy")
    print("\nSaved: results\\week7_tradeoff.png and results\\week7_tradeoff.csv")


if __name__ == "__main__":
    main()
