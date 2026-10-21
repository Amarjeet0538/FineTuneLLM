"""
split_data.py - Week 2, step 1.

Splits records_labeled.jsonl into train / dev / eval BY TEMPLATE and writes
data/split/eval_review.csv for you to hand-check. Nothing is frozen yet;
freeze_eval.py does that after your review.

Eval has two parts:
  unseen_template : templates the model never sees in training (the honest test)
  seen_template   : new lines of templates that ARE in training (in-distribution)
"""
import csv, random, sys
from collections import Counter, defaultdict
import config as C

REQUIRED = ["system", "input", "service", "event_type", "severity",
            "identifiers", "recommended_action"]


def die(msg):
    print("\nSTOPPED: " + msg)
    sys.exit(1)


def main():
    if C.EVAL_FROZEN.exists():
        die(f"{C.EVAL_FROZEN.name} already exists - the eval set is frozen. "
            "Re-splitting now would break the freeze. Nothing was changed.")

    recs = C.read_jsonl(C.RECORDS)
    for i, r in enumerate(recs):
        missing = [k for k in REQUIRED if k not in r]
        if missing:
            die(f"record {i} is missing fields {missing}")
        r["id"] = f"r{i:05d}"
        r["_action_original"] = r["recommended_action"]
        r["recommended_action"] = C.ACTION_MERGE.get(r["recommended_action"],
                                                     r["recommended_action"])

    # --- sanity checks on the label vocabulary ---
    bad_sev = Counter(r["severity"] for r in recs if r["severity"] not in C.SEVERITIES)
    if bad_sev:
        die(f"severity values not in {C.SEVERITIES}: {dict(bad_sev)}")
    bad_act = Counter(r["recommended_action"] for r in recs
                      if r["recommended_action"] not in C.ACTIONS)
    if bad_act:
        die(f"actions not in ACTIONS list (after merge): {dict(bad_act)}. "
            "Add them to ACTIONS or ACTION_MERGE in config.py.")

    # --- one entry per template ---
    by_t = defaultdict(list)
    for r in recs:
        by_t[r["event_type"]].append(r)
    t_action = {}
    mixed = 0
    for t, rs in by_t.items():
        c = Counter(r["recommended_action"] for r in rs)
        t_action[t] = c.most_common(1)[0][0]
        mixed += len(c) > 1
    per_action = defaultdict(list)
    for t in sorted(by_t):
        per_action[t_action[t]].append(t)

    print(f"{len(recs)} records, {len(by_t)} templates "
          f"({mixed} templates have >1 action across severities; majority used for stratifying)")
    print("\nTemplates per action (after merge):")
    for a in C.ACTIONS:
        print(f"  {a:16s} {len(per_action[a]):4d}")
    thin = [a for a in C.ACTIONS if len(per_action[a]) < C.MIN_TEMPLATES_PER_ACTION]
    if thin:
        die(f"{thin} have fewer than {C.MIN_TEMPLATES_PER_ACTION} templates - too few to "
            "train AND evaluate on. Add them to ACTION_MERGE (and remove from ACTIONS) in config.py.")

    # --- choose held-out templates, stratified by action ---
    rng = random.Random(C.SEED)
    eval_t, dev_t = set(), set()
    for a in C.ACTIONS:
        ts = per_action[a][:]
        rng.shuffle(ts)
        n_eval = max(1, int(len(ts) * C.EVAL_UNSEEN_FRACTION + 0.5))
        n_dev = max(1, int(len(ts) * C.DEV_UNSEEN_FRACTION + 0.5))
        eval_t.update(ts[:n_eval])
        dev_t.update(ts[n_eval:n_eval + n_dev])

    # --- assign every line to a bucket ---
    # Work with DISTINCT message texts: LogHub repeats many lines word-for-word,
    # and an eval line must never have an identical twin in train.
    for t in sorted(by_t):
        by_text = defaultdict(list)
        for r in by_t[t]:
            by_text[r["input"]].append(r)
        texts = sorted(by_text)
        rng.shuffle(texts)

        if t in eval_t or t in dev_t:
            tag = "eval" if t in eval_t else "dev"
            for j, txt in enumerate(texts):
                for k, r in enumerate(by_text[txt]):
                    if j < C.UNSEEN_LINES_CAP and k == 0:
                        r["split"], r["eval_kind"] = tag, "unseen_template"
                    else:
                        r["split"] = "discard"  # held-out template: never in train
            continue

        # training template: carve out "seen" eval/dev lines, train keeps >= 1 distinct text
        n_seen = min(C.SEEN_LINES_PER_TEMPLATE, len(texts) - 1)
        n_dev = 1 if len(texts) - n_seen >= 2 else 0
        for j, txt in enumerate(texts):
            for k, r in enumerate(by_text[txt]):
                if j < n_seen:
                    tag = "eval"
                elif j < n_seen + n_dev:
                    tag = "dev"
                else:
                    r["split"] = "train"
                    continue
                if k == 0:
                    r["split"], r["eval_kind"] = tag, "seen_template"
                else:
                    r["split"] = "discard"      # identical copy of a held-out line

    # hard guarantees - fail loudly rather than silently leak
    train_text = {r["input"] for r in recs if r["split"] == "train"}
    train_t = {r["event_type"] for r in recs if r["split"] == "train"}
    for r in recs:
        if r["split"] in ("eval", "dev"):
            assert r["input"] not in train_text, f"leak: {r['id']} text is in train"
            if r["eval_kind"] == "unseen_template":
                assert r["event_type"] not in train_t, f"leak: {r['event_type']} in train"
            else:
                assert r["event_type"] in train_t, f"{r['event_type']} seen but not in train"

    C.write_jsonl(C.STAGING, recs)

    # --- report ---
    ev = [r for r in recs if r["split"] == "eval"]
    print(f"\nLines per bucket: {dict(Counter(r['split'] for r in recs))}")
    print(f"\nEVAL = {len(ev)} lines")
    print(f"  by kind   : {dict(Counter(r['eval_kind'] for r in ev))}")
    print(f"  by system : {dict(Counter(r['system'] for r in ev))}")
    print(f"  by action : {dict(Counter(r['recommended_action'] for r in ev))}")
    print(f"  unseen templates per system: "
          f"{dict(Counter(t.split(':')[0] for t in eval_t))}")
    if not 300 <= len(ev) <= 500:
        print(f"  NOTE: eval size {len(ev)} is outside the planned 300-500.")

    # --- review sheet: ONE row per (template, severity) = the unit the teacher labelled ---
    ev_g = defaultdict(list)
    for r in ev:
        ev_g[(r["event_type"], r["severity"])].append(r)
    with open(C.REVIEW_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event_type", "severity", "eval_kind", "n_eval_lines", "service",
                    "template", "example_line", "recommended_action", "verified"])
        for g in sorted(ev_g, key=lambda x: (ev_g[x][0]["eval_kind"], x)):
            rs = ev_g[g]
            w.writerow([g[0], g[1], rs[0]["eval_kind"], len(rs), rs[0]["service"],
                        rs[0]["_event_template"], rs[0]["input"],
                        rs[0]["recommended_action"], ""])
    print(f"\nWrote {C.STAGING}")
    print(f"Wrote {C.REVIEW_CSV}  ({len(ev_g)} rows to review)")
    print("Next: review that CSV, then run freeze_eval.py")


if __name__ == "__main__":
    main()
