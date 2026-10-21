"""
build_train_set.py - Week 3, step 1.

Turns data/split/train.jsonl and data/split/dev.jsonl into the exact
prompt/answer pairs used for fine-tuning on Kaggle:

    data/week3/train_sft.jsonl   (what the model learns from)
    data/week3/dev_sft.jsonl     (only used to pick the best checkpoint)
    data/week3/build_report.json (numbers for the report)

    prompt = config.build_prompt(log line)   <- the SAME frozen prompt used for scoring
    answer = config.format_target(record)    <- the Kaggle notebook adds <|im_end|> after it

Training-set rules (fixed here, before any training):
  1. Exact duplicates removed: the same log line with the same answer counts once.
  2. Each no_action (template, severity) group gives at most CAP_PER_GROUP
     examples, picked at random with the project seed. Stops one pattern with
     900+ lines from drowning out the rare ones. Groups of the rare actions
     (investigate, escalate_oncall, check_network) are never capped: every
     distinct example of them is kept.
Dev is kept as-is (every line). The eval set is read only to double-check
that none of its log lines ended up in train or dev.
"""
import json, random, sys
from collections import Counter, defaultdict
import config as C

CAP_PER_GROUP = 20
CAPPED_ACTIONS = {"no_action"}      # only the majority class is thinned out
OUT_DIR = C.ROOT / "data" / "week3"
TRAIN_OUT = OUT_DIR / "train_sft.jsonl"
DEV_OUT = OUT_DIR / "dev_sft.jsonl"
REPORT = OUT_DIR / "build_report.json"


def die(msg):
    print("\nSTOPPED: " + msg + "\nNothing was written.")
    sys.exit(1)


def pair(r):
    return {"id": r["id"], "event_type": r["event_type"], "severity": r["severity"],
            "action": r["recommended_action"], "system": r["system"],
            "eval_kind": r.get("eval_kind", ""),
            "prompt": C.build_prompt(r["input"]), "answer": C.format_target(r)}


def main():
    C.verify_frozen()                       # stops if eval was changed after freezing
    train = C.read_jsonl(C.TRAIN)
    dev = C.read_jsonl(C.DEV)
    eval_texts = {r["input"] for r in C.read_jsonl(C.EVAL_FROZEN)}

    # --- safety: no eval line may be in train or dev ---
    leaks = [r["id"] for r in train + dev if r["input"] in eval_texts]
    if leaks:
        die(f"{len(leaks)} train/dev lines also appear in the frozen eval set, e.g. {leaks[:3]}")
    for r in train + dev:
        if r["recommended_action"] not in C.ACTIONS or r["severity"] not in C.SEVERITIES:
            die(f"record {r['id']} has a label outside the allowed lists")

    # --- 1. remove exact duplicates (same line + same answer) ---
    uniq, answers_per_text = {}, defaultdict(set)
    for r in train:
        answers_per_text[r["input"]].add(C.format_target(r))
        uniq.setdefault((r["input"], C.format_target(r)), r)
    conflicting = sum(1 for a in answers_per_text.values() if len(a) > 1)

    # --- 2. cap each no_action (template, severity) group ---
    groups = defaultdict(list)
    for r in uniq.values():
        groups[(r["event_type"], r["severity"])].append(r)
    rng = random.Random(C.SEED)
    kept = []
    for g in sorted(groups):
        rows = sorted(groups[g], key=lambda r: r["id"])
        rng.shuffle(rows)
        capped = rows[0]["recommended_action"] in CAPPED_ACTIONS
        kept.extend(rows[:CAP_PER_GROUP] if capped else rows)
    rng.shuffle(kept)

    train_pairs = [pair(r) for r in kept]
    dev_pairs = [pair(r) for r in dev]
    C.write_jsonl(TRAIN_OUT, train_pairs)
    C.write_jsonl(DEV_OUT, dev_pairs)

    report = {
        "cap_per_group": CAP_PER_GROUP,
        "capped_actions": sorted(CAPPED_ACTIONS),
        "seed": C.SEED,
        "train_lines_in": len(train),
        "train_distinct_pairs": len(uniq),
        "train_groups": len(groups),
        "texts_with_more_than_one_answer": conflicting,
        "train_examples_out": len(train_pairs),
        "dev_examples_out": len(dev_pairs),
        "train_by_action_before": dict(Counter(r["recommended_action"] for r in train)),
        "train_by_action_deduplicated": dict(Counter(r["recommended_action"] for r in uniq.values())),
        "train_by_action_after": dict(Counter(p["action"] for p in train_pairs)),
        "train_by_system_after": dict(Counter(p["system"] for p in train_pairs)),
        "dev_by_action": dict(Counter(p["action"] for p in dev_pairs)),
        "dev_by_kind": dict(Counter(p["eval_kind"] for p in dev_pairs)),
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"train: {len(train)} lines -> {len(uniq)} after removing duplicates "
          f"-> {len(train_pairs)} after capping each no_action group at {CAP_PER_GROUP}")
    print(f"   by action, all lines        : {report['train_by_action_before']}")
    print(f"   by action, duplicates gone  : {report['train_by_action_deduplicated']}")
    print(f"   by action, after the cap    : {report['train_by_action_after']}")
    print(f"   by system, after the cap    : {report['train_by_system_after']}")
    print(f"dev  : {len(dev_pairs)} examples  {report['dev_by_kind']}")
    if conflicting:
        print(f"NOTE: {conflicting} log lines appear in train with more than one answer "
              "(same text, different severity/label).\n      Expected: severity comes from the "
              "log header the model never sees. Both versions are kept; mention it in the report.")
    print(f"\nWrote {TRAIN_OUT}\nWrote {DEV_OUT}\nWrote {REPORT}")
    print("Next: upload train_sft.jsonl and dev_sft.jsonl to Kaggle as a private dataset.")


if __name__ == "__main__":
    main()