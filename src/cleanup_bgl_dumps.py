"""
cleanup_bgl_dumps.py — one-shot fix.

BGL logs its post-crash register/status dump line-by-line, and every line
inherits the crash's FATAL severity. Those dump fragments (register readouts,
status bits) are not independently actionable, but the teacher labeled several
of them escalate_oncall / check_disk. This reclassifies that whole family to
no_action, leaving genuine crash EVENTS (rts panic, kernel terminated, machine
check, Lustre mount failed, etc.) untouched.

Run from the project root:  python src/cleanup_bgl_dumps.py
"""
import json, os, re, csv

DATA = os.path.join("data", "labeled")

EXPLICIT_REG = {
 "special purpose registers:", "machine state register: <*>", "Machine State Register: <*>",
 "exception syndrome register: <*>", "core configuration register: <*>",
 "data address: <*>", "instruction address: <*>", "fpr29=<*>",
 "r24=<*> r25=<*> r26=<*> r27=<*>", "lr:<*> cr:<*> xer:<*> ctr:<*>",
 "dbcr0=<*> dbsr=<*> ccr0=<*>", "iar <*> dear <*>",
}
def is_dump(t):
    t = t or ""
    if re.search(r"\.\.+", t): return True          # dotted-leader readout
    if re.search(r"\w\.<\*>\s*$", t): return True    # single-dot leader at end
    if t in EXPLICIT_REG: return True
    return False

# 1) load + dedupe the per-group cache (keep last decision per group)
cache_path = os.path.join(DATA, "action_cache.jsonl")
by_group = {}
for line in open(cache_path, encoding="utf-8"):
    d = json.loads(line)
    by_group[d["group"]] = d
cache = list(by_group.values())

# 2) reclassify BGL dump groups -> no_action
changed = 0
for d in cache:
    if d.get("event_type","").startswith("BGL:") and is_dump(d.get("event_template","")):
        if d.get("recommended_action") != "no_action":
            d["recommended_action"] = "no_action"
            d["reason"] = "reclassified: BGL RAS register/status dump line, not independently actionable"
            changed += 1

# write cache back (deduped + fixed)
with open(cache_path, "w", encoding="utf-8") as f:
    for d in cache:
        f.write(json.dumps(d) + "\n")

# 3) rebuild records_labeled.jsonl from records_partial + fixed cache
action_by_group = {d["group"]: d["recommended_action"] for d in cache}
recs = [json.loads(l) for l in open(os.path.join(DATA,"records_partial.jsonl"), encoding="utf-8")]
def gk(r): return f'{r["event_type"]}||{r["severity"]}'
unresolved = 0
with open(os.path.join(DATA,"records_labeled.jsonl"), "w", encoding="utf-8") as f:
    for r in recs:
        a = action_by_group.get(gk(r), "NEEDS_REVIEW")
        r["recommended_action"] = a
        unresolved += (a == "NEEDS_REVIEW")
        f.write(json.dumps(r) + "\n")

# 4) rebuild groups_for_review.csv (utf-8, so fancy dashes don't crash it)
with open(os.path.join(DATA,"groups_for_review.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["event_type","severity","service","recommended_action","reason","event_template","example"])
    for d in sorted(cache, key=lambda x:(x["recommended_action"], x["event_type"])):
        w.writerow([d["event_type"],d["severity"],d["service"],d["recommended_action"],
                    d["reason"],d["event_template"],d.get("example","")])

print(f"Reclassified {changed} BGL dump groups -> no_action")
print(f"Records unresolved: {unresolved}/{len(recs)}")
print("Rewrote: action_cache.jsonl, records_labeled.jsonl, groups_for_review.csv")
