"""
freeze_eval.py - Week 2, step 2 (run ONCE, after reviewing eval_review.csv).

In the CSV, for every row:
  verified            -> y     (checked, label is right, or you corrected it)
                      -> drop  (bad/ambiguous example, leave it out of eval)
  recommended_action  -> change it here if the teacher got it wrong.
                         A correction is applied to that (template, severity)
                         group everywhere - eval, dev AND train - so the
                         training labels agree with the answer key.
Only those two columns are read back; everything else comes from staging, so
Excel can't corrupt the log text.
"""
import csv, json, os, stat, sys
from collections import Counter
from datetime import datetime, timezone
import config as C


def die(msg):
    print("\nSTOPPED: " + msg + "\nNothing was written.")
    sys.exit(1)


def read_review():
    for enc in ("utf-8-sig", "cp1252"):   # Excel may save as either
        try:
            with open(C.REVIEW_CSV, encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    die("could not read eval_review.csv - save it as 'CSV UTF-8' from Excel")


def main():
    if C.EVAL_FROZEN.exists():
        die("eval_frozen.jsonl already exists. The eval set is frozen.")
    if not C.STAGING.exists():
        die("no staging.jsonl - run split_data.py first")

    recs = C.read_jsonl(C.STAGING)
    eval_groups = {(r["event_type"], r["severity"]) for r in recs if r["split"] == "eval"}

    rows = read_review()
    seen, problems, changes, drops = set(), [], {}, set()
    for i, row in enumerate(rows, start=2):           # row 1 is the header
        g = (row.get("event_type", "").strip(), row.get("severity", "").strip())
        v = row.get("verified", "").strip().lower()
        a = row.get("recommended_action", "").strip()
        if g not in eval_groups:
            problems.append(f"line {i}: {g} is not an eval group (edited event_type/severity?)")
            continue
        if g in seen:
            problems.append(f"line {i}: {g} appears twice")
        seen.add(g)
        if v not in ("y", "drop"):
            problems.append(f"line {i}: verified must be y or drop, got '{v}'")
        if v == "drop":
            drops.add(g)
        elif a not in C.ACTIONS:
            problems.append(f"line {i}: action '{a}' not in {C.ACTIONS}")
        else:
            current = next(r["recommended_action"] for r in recs
                           if (r["event_type"], r["severity"]) == g)
            if a != current:
                changes[g] = (current, a)
    missing = eval_groups - seen
    if missing:
        problems.append(f"{len(missing)} eval groups missing from the CSV, e.g. {sorted(missing)[:3]}")
    if problems:
        die("fix these in eval_review.csv:\n  " + "\n  ".join(problems[:30]))

    # apply corrections to the whole group (eval + dev + train), drop rejected eval lines
    for r in recs:
        g = (r["event_type"], r["severity"])
        if g in changes:
            r["recommended_action"] = changes[g][1]
        if r["split"] == "eval" and g in drops:
            r["split"] = "dropped_in_review"

    def clean(r):
        return {k: v for k, v in r.items() if k != "split"}

    train = [clean(r) for r in recs if r["split"] == "train"]
    dev = [clean(r) for r in recs if r["split"] == "dev"]
    ev = [clean(r) for r in recs if r["split"] == "eval"]

    C.write_jsonl(C.TRAIN, train)
    C.write_jsonl(C.DEV, dev)
    C.write_jsonl(C.EVAL_FROZEN, ev)
    digest = C.sha256_of(C.EVAL_FROZEN)
    C.EVAL_HASH.write_text(f"{digest}  eval_frozen.jsonl\n", encoding="utf-8")
    os.chmod(C.EVAL_FROZEN, stat.S_IREAD)          # read-only: accidental edits fail

    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "eval_sha256": digest,
        "seed": C.SEED,
        "settings": {k: getattr(C, k) for k in [
            "EVAL_UNSEEN_FRACTION", "DEV_UNSEEN_FRACTION", "UNSEEN_LINES_CAP",
            "SEEN_LINES_PER_TEMPLATE", "ACTION_MERGE", "ACTIONS", "SEVERITIES", "OUTPUT_KEYS"]},
        "instructions": C.INSTRUCTIONS,
        "counts": {"train": len(train), "dev": len(dev), "eval": len(ev)},
        "eval_by_kind": dict(Counter(r["eval_kind"] for r in ev)),
        "eval_by_action": dict(Counter(r["recommended_action"] for r in ev)),
        "review_changes": {f"{g[0]}|{g[1]}": list(v) for g, v in changes.items()},
        "review_drops": sorted(f"{g[0]}|{g[1]}" for g in drops),
    }
    C.MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"train {len(train)} | dev {len(dev)} | eval {len(ev)}")
    print(f"eval by kind  : {manifest['eval_by_kind']}")
    print(f"eval by action: {manifest['eval_by_action']}")
    print(f"labels corrected in review: {len(changes)} groups; dropped: {len(drops)} groups")
    for g, (old, new) in changes.items():
        print(f"   {g[0]} {g[1]}: {old} -> {new}")
    print(f"\nFROZEN. sha256 = {digest}")
    print(f"Wrote {C.EVAL_FROZEN} (read-only), {C.EVAL_HASH}, {C.TRAIN}, {C.DEV}, {C.MANIFEST}")


if __name__ == "__main__":
    main()
