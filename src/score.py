"""
score.py - scores a predictions file against the frozen eval set (or dev).

    python src\\score.py results\\preds_eval_<name>.jsonl
    python src\\score.py results\\preds_dev_<name>.jsonl --split dev

Rules (fixed now, before any model is scored):
  * Every eval example counts. Unparseable / missing output = wrong on every field.
  * Parsing: strict = the whole output is one JSON object. Lenient (used for the
    field scores) = drop any <think>...</think> and ``` fences, then parse the
    text from the first "{" to the last "}".
  * schema_valid = all 4 keys present, strings where strings are expected,
    identifiers is a list of strings. Extra keys are ignored.
  * Categorical fields: exact match after trimming and ignoring upper/lower case.
  * Identifiers: exact string match (trimmed), order ignored, duplicates counted.
    Precision/recall/F1 are micro-averaged over all examples.
  * recommended_action macro-F1 averages over the classes present in the gold labels.
  * 95% confidence intervals: bootstrap, 1000 resamples, fixed seed.
"""
import argparse, json, random, re
from collections import Counter
from pathlib import Path
import config as C

THINK = re.compile(r"<think>.*?</think>", re.S)
FENCE = re.compile(r"```(?:json)?", re.I)
CAT_FIELDS = ["service", "severity", "recommended_action"]


def parse(raw):
    raw = raw or ""
    try:
        strict = isinstance(json.loads(raw.strip()), dict)
    except Exception:
        strict = False
    txt = FENCE.sub("", THINK.sub("", raw))
    i, j = txt.find("{"), txt.rfind("}")
    obj = None
    if i != -1 and j > i:
        try:
            o = json.loads(txt[i:j + 1])
            obj = o if isinstance(o, dict) else None
        except Exception:
            pass
    return strict, obj


def norm(x):
    return x.strip().lower() if isinstance(x, str) else None


def score_one(gold, raw):
    strict, o = parse(raw)
    o = o or {}
    schema = (all(k in o for k in C.OUTPUT_KEYS)
              and all(isinstance(o[k], str) for k in CAT_FIELDS)
              and isinstance(o["identifiers"], list)
              and all(isinstance(x, str) for x in o["identifiers"]))
    ids_p = o.get("identifiers")
    ids_p = Counter(str(x).strip() for x in ids_p) if isinstance(ids_p, list) else Counter()
    ids_g = Counter(x.strip() for x in gold["identifiers"])
    act_p = norm(o.get("recommended_action"))
    return {
        "parsed": bool(o), "strict": strict, "schema": schema,
        "vocab": (norm(o.get("severity")) in {x.lower() for x in C.SEVERITIES}
                  and act_p in {x.lower() for x in C.ACTIONS}),
        **{f"{f}_ok": norm(o.get(f)) == norm(gold[f]) for f in CAT_FIELDS},
        "act_p": act_p, "act_g": norm(gold["recommended_action"]),
        "id_tp": sum((ids_p & ids_g).values()), "id_np": sum(ids_p.values()),
        "id_ng": sum(ids_g.values()), "id_exact": ids_p == ids_g,
    }


def metrics(rows):
    n = len(rows)
    if n == 0:
        return {}
    m = {"n": n}
    for k in ["parsed", "strict", "schema", "vocab"]:
        m[k + "_rate"] = sum(r[k] for r in rows) / n
    for f in CAT_FIELDS:
        m[f + "_acc"] = sum(r[f + "_ok"] for r in rows) / n
    tp, npred, ngold = (sum(r[k] for r in rows) for k in ("id_tp", "id_np", "id_ng"))
    p = tp / npred if npred else 0.0
    rc = tp / ngold if ngold else 0.0
    m["id_precision"], m["id_recall"] = p, rc
    m["id_f1"] = 2 * p * rc / (p + rc) if p + rc else 0.0
    m["id_exact_rate"] = sum(r["id_exact"] for r in rows) / n
    f1s = []
    for c in sorted({r["act_g"] for r in rows}):
        tpc = sum(r["act_p"] == c and r["act_g"] == c for r in rows)
        pc = sum(r["act_p"] == c for r in rows)
        gc = sum(r["act_g"] == c for r in rows)
        pr, re_ = (tpc / pc if pc else 0.0), tpc / gc
        f1s.append(2 * pr * re_ / (pr + re_) if pr + re_ else 0.0)
    m["action_macro_f1"] = sum(f1s) / len(f1s)
    return m


CI_KEYS = ["schema_rate", "service_acc", "severity_acc", "recommended_action_acc",
           "action_macro_f1", "id_f1"]


def with_ci(rows, b=1000):
    point = metrics(rows)
    rng = random.Random(0)
    boots = [metrics([rows[rng.randrange(len(rows))] for _ in rows]) for _ in range(b)]
    ci = {}
    for k in CI_KEYS:
        v = sorted(x[k] for x in boots)
        ci[k] = (v[int(0.025 * b)], v[int(0.975 * b) - 1])
    return point, ci


def per_class(rows):
    out = {}
    for c in sorted({r["act_g"] for r in rows} | {r["act_p"] for r in rows if r["act_p"] in {x.lower() for x in C.ACTIONS}}):
        tp = sum(r["act_p"] == c and r["act_g"] == c for r in rows)
        pc = sum(r["act_p"] == c for r in rows)
        gc = sum(r["act_g"] == c for r in rows)
        p, rc = (tp / pc if pc else 0.0), (tp / gc if gc else 0.0)
        out[c] = {"support": gc, "precision": p, "recall": rc,
                  "f1": 2 * p * rc / (p + rc) if p + rc else 0.0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds")
    ap.add_argument("--split", choices=["eval", "dev"], default="eval")
    args = ap.parse_args()

    if args.split == "eval":
        C.verify_frozen()
        gold = C.read_jsonl(C.EVAL_FROZEN)
    else:
        gold = C.read_jsonl(C.DEV)
    preds = {p["id"]: p for p in C.read_jsonl(args.preds)}
    missing = [g["id"] for g in gold if g["id"] not in preds]
    if missing:
        print(f"WARNING: {len(missing)} of {len(gold)} examples have no prediction "
              "- they are scored as wrong. Finish the run before reporting numbers.")
    truncated = sum(1 for p in preds.values() if p.get("done_reason") == "length")

    rows = []
    for g in gold:
        r = score_one(g, preds.get(g["id"], {}).get("raw_output", ""))
        r["kind"], r["system"] = g.get("eval_kind", "?"), g["system"]
        rows.append(r)

    report = {"preds_file": str(args.preds), "split": args.split,
              "missing": len(missing), "truncated_outputs": truncated}
    subsets = [("ALL", rows)] + [(k, [r for r in rows if r["kind"] == k])
                                 for k in ("seen_template", "unseen_template")]
    print(f"\nScoring {args.preds}  ({args.split}, n={len(rows)}, truncated={truncated})")
    for name, sub in subsets:
        if not sub:
            continue
        point, ci = with_ci(sub)
        report[name] = {"metrics": point, "ci95": ci}
        print(f"\n== {name}  (n={point['n']})")
        print(f"   parsed {point['parsed_rate']:.1%} | strict JSON {point['strict_rate']:.1%} | "
              f"schema-valid {point['schema_rate']:.1%} | allowed values {point['vocab_rate']:.1%}")
        for k in CI_KEYS[1:]:
            lo, hi = ci[k]
            print(f"   {k:24s} {point[k]:6.1%}   [95% CI {lo:.1%} - {hi:.1%}]")
        print(f"   {'id_precision / recall':24s} {point['id_precision']:6.1%} / {point['id_recall']:.1%}"
              f"   | identifiers exactly right: {point['id_exact_rate']:.1%}")

    report["action_per_class"] = per_class(rows)
    print("\n== recommended_action per class (ALL)")
    for c, v in report["action_per_class"].items():
        print(f"   {c:16s} support {v['support']:4d}  P {v['precision']:.2f}  "
              f"R {v['recall']:.2f}  F1 {v['f1']:.2f}")
    report["by_system"] = {s: metrics([r for r in rows if r["system"] == s])
                           for s in sorted({r["system"] for r in rows})}
    print("\n== by system  (action acc | severity acc | service acc | id F1)")
    for s, m in report["by_system"].items():
        print(f"   {s:10s} n={m['n']:4d}  {m['recommended_action_acc']:.1%} | "
              f"{m['severity_acc']:.1%} | {m['service_acc']:.1%} | {m['id_f1']:.1%}")

    C.RESULTS.mkdir(exist_ok=True)
    out = C.RESULTS / (Path(args.preds).stem.replace("preds_", "scores_") + ".json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
