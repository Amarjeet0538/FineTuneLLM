"""
label_with_teacher.py — Week 1/2 teacher distillation for `recommended_action`.

Only ONE schema field is teacher-generated: recommended_action (closed vocab).
service/event_type/severity/identifiers are ground truth from build_dataset.py.

Key design:
  * PER-GROUP labeling. recommended_action depends on (event_type, severity),
    not on which identifiers appear. We label one representative per group
    (~222 calls) and propagate to all 8000 records. 36x fewer calls, consistent
    labels, and the human only hand-verifies ~222 decisions (groups_for_review.csv).
  * Closed vocabulary, enforced + repaired. Off-vocab -> NEEDS_REVIEW, never guessed.
  * Resumable: group decisions cached to action_cache.jsonl; reruns skip done groups.
  * Providers: mock (tested here) | groq | gemini (SDK calls verified against
    installed groq==1.7.0 / google-genai==2.23.0; MODEL NAME + a live key are yours).

Run order for the owner:
  1) python src/label_with_teacher.py --provider mock      # plumbing sanity, NOT for training
  2) python src/label_with_teacher.py --provider groq --model <current-free-model>
  3) hand-verify data/labeled/groups_for_review.csv, fix any NEEDS_REVIEW, re-run to propagate
"""
import json, os, sys, time, argparse, re

VOCAB = {
    "no_action":       "routine/informational; nothing to do",
    "investigate":     "anomaly or error of unclear cause; needs a human to look",
    "escalate_oncall": "severe/critical impact; page the on-call engineer",
    "restart_service": "a process/service is down or stuck and should be restarted",
    "check_disk":      "storage/block/filesystem/space problem",
    "check_network":   "connectivity/socket/timeout/host-unreachable problem",
    "check_auth":      "authentication/authorization/permission/credential problem",
}
SYNONYMS = {  # cheap repair for near-miss teacher outputs
    "none": "no_action", "ignore": "no_action", "ok": "no_action",
    "escalate": "escalate_oncall", "page_oncall": "escalate_oncall", "page": "escalate_oncall",
    "restart": "restart_service", "reboot": "restart_service",
    "disk": "check_disk", "storage": "check_disk",
    "network": "check_network", "connectivity": "check_network",
    "auth": "check_auth", "authentication": "check_auth", "permission": "check_auth",
    "look": "investigate", "review": "investigate", "check": "investigate",
}
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "labeled")
CACHE = os.path.join(DATA, "action_cache.jsonl")

def group_key(r): return f"{r['event_type']}||{r['severity']}"

def build_prompt(rep):
    vocab_lines = "\n".join(f"  - {k}: {v}" for k, v in VOCAB.items())
    return (
        "You triage software logs. Pick exactly ONE recommended_action from this closed list:\n"
        f"{vocab_lines}\n\n"
        "Log group:\n"
        f"  severity      : {rep['severity']}\n"
        f"  service       : {rep['service']}\n"
        f"  event_template: {rep['_event_template']}\n"
        f"  example_line  : {rep['input']}\n\n"
        'Reply ONLY with JSON: {"recommended_action": "<one value from the list>", '
        '"reason": "<max 15 words>"}'
    )

# ---------------- providers ----------------
def call_mock(rep):
    """Deterministic heuristic. Proves the pipeline. NOT valid training data."""
    t = (rep["_event_template"] + " " + rep["input"]).lower()
    sev = rep["severity"]
    def has(*ws): return any(w in t for w in ws)
    if has("auth", "permission", "denied", "credential", "login", "token"): a="check_auth"
    elif has("disk", "block", "storage", "space", "filesystem", "quota", "blk_"): a="check_disk"
    elif has("connection","socket","timeout","unreachable","network","refused","10."): a="check_network"
    elif has("restart","terminating","shutdown","killed","stopped","exit"): a="restart_service"
    elif sev in ("FATAL","SEVERE"): a="escalate_oncall"
    elif has("exception","error","fail","corrupt","fatal") or sev in ("ERROR","WARNING"): a="investigate"
    else: a="no_action"
    return json.dumps({"recommended_action": a, "reason": "mock-heuristic"})

def call_groq(rep, model, key):
    from groq import Groq
    c = Groq(api_key=key)
    r = c.chat.completions.create(
        model=model, temperature=0, max_tokens=1024,
        messages=[{"role": "user", "content": build_prompt(rep)}],
    )
    return r.choices[0].message.content

def call_gemini(rep, model, key):
    from google import genai
    from google.genai import types
    c = genai.Client(api_key=key)
    r = c.models.generate_content(
        model=model, contents=build_prompt(rep),
        config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"),
    )
    return r.text

# ---------------- validation ----------------
def normalize(raw_json):
    import re as _re
    try:
        _m = _re.search(r"\{.*\}", raw_json or "", _re.DOTALL)
        obj = json.loads(_m.group(0) if _m else (raw_json or ""))
        a = str(obj.get("recommended_action", "")).strip().lower().replace(" ", "_")
        reason = str(obj.get("reason", ""))[:80]
    except Exception:
        return "NEEDS_REVIEW", f"unparseable: {raw_json[:60]}"
    if a in VOCAB: return a, reason
    if a in SYNONYMS: return SYNONYMS[a], reason + " [repaired]"
    return "NEEDS_REVIEW", f"off-vocab: {a} | {reason}"

def load_cache():
    done = {}
    if os.path.exists(CACHE):
        for l in open(CACHE):
            d = json.loads(l); done[d["group"]] = d
    return done

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["mock","groq","gemini"], default="mock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(os.path.join(DATA, "records_partial.jsonl"))]
    groups = {}
    for r in recs:
        groups.setdefault(group_key(r), r)  # first record = representative

    key = os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if args.provider != "mock" and (not key or not args.model):
        sys.exit(f"[{args.provider}] needs env {args.provider.upper()}_API_KEY and --model <name>")

    done = load_cache()
    cache_f = open(CACHE, "a")
    labeled_groups, review = 0, 0
    for i, (g, rep) in enumerate(groups.items(), 1):
        if g in done: continue
        for attempt in range(3):
            try:
                raw = {"mock":call_mock,"groq":lambda r:call_groq(r,args.model,key),
                       "gemini":lambda r:call_gemini(r,args.model,key)}[args.provider](rep)
                break
            except Exception as e:
                if attempt == 2: raw = json.dumps({"recommended_action":"NEEDS_REVIEW","reason":f"api_error:{e}"})
                else: time.sleep(2**attempt)
        action, reason = normalize(raw)
        rec = {"group": g, "event_type": rep["event_type"], "severity": rep["severity"],
               "service": rep["service"], "event_template": rep["_event_template"],
               "example": rep["input"][:100], "recommended_action": action, "reason": reason}
        cache_f.write(json.dumps(rec) + "\n"); cache_f.flush()
        done[g] = rec
        labeled_groups += 1
        review += (action == "NEEDS_REVIEW")
        if args.sleep: time.sleep(args.sleep)
    cache_f.close()

    # propagate to all 8000
    out = os.path.join(DATA, "records_labeled.jsonl")
    nulls = 0
    with open(out, "w") as f:
        for r in recs:
            a = done[group_key(r)]["recommended_action"]
            r["recommended_action"] = a
            nulls += (a == "NEEDS_REVIEW")
            f.write(json.dumps(r) + "\n")

    # human-verification sheet
    import csv
    with open(os.path.join(DATA, "groups_for_review.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["event_type","severity","service","recommended_action","reason","event_template","example"])
        for d in sorted(done.values(), key=lambda x:(x["recommended_action"], x["event_type"])):
            w.writerow([d["event_type"],d["severity"],d["service"],d["recommended_action"],d["reason"],d["event_template"],d["example"]])

    print(f"provider={args.provider}  groups labeled this run={labeled_groups}  total groups={len(groups)}")
    print(f"NEEDS_REVIEW groups={sum(1 for d in done.values() if d['recommended_action']=='NEEDS_REVIEW')}")
    print(f"records with unresolved action={nulls}/{len(recs)}")
    print("wrote: records_labeled.jsonl, groups_for_review.csv, action_cache.jsonl")

if __name__ == "__main__":
    main()
