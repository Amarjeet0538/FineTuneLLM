"""
build_dataset.py — Week 1 parsing + ground-truth field mapping.

Reads LogHub-2k structured CSVs (which ALREADY carry hand-corrected
EventTemplate + EventId) and assembles partial schema records.

Ground-truth fields (NO teacher needed):
    service      <- Component
    event_type   <- "<system>:<EventId>"   (namespaced; EventId is opaque + system-local)
    severity     <- normalised Level
    identifiers  <- aligned from Content against the verified EventTemplate

Teacher-only field (added later by label_with_teacher.py):
    recommended_action

Drain3 is deliberately NOT used: on the 2k sets it matches the corrected
ground-truth template 0% of the time and would inject parser error.
"""
import pandas as pd, re, json, os

SYSTEMS = ["HDFS", "BGL", "OpenStack", "Spark"]
RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "labeled")

# --- severity normalisation (flag: SEVERE mapping is a judgment call) ---
SEV_MAP = {
    "INFO": "INFO", "WARN": "WARNING", "WARNING": "WARNING",
    "ERROR": "ERROR", "FATAL": "FATAL", "SEVERE": "ERROR",  # <-- confirm this
}

def template_to_regex(template):
    parts = str(template).split("<*>")
    return "^" + "(.*?)".join(re.escape(p) for p in parts) + r"\s*$"

def extract_identifiers(content, template):
    m = re.match(template_to_regex(template), str(content).strip())
    if not m:
        return None
    return [g.strip() for g in m.groups() if g.strip() != ""]

def build_system(sys):
    df = pd.read_csv(os.path.join(RAW, f"{sys}_2k.log_structured.csv"))
    recs, fails = [], 0
    for _, r in df.iterrows():
        ids = extract_identifiers(r["Content"], r["EventTemplate"])
        if ids is None:
            fails += 1
            continue
        raw_level = str(r["Level"]).upper() if "Level" in df.columns else "NA"
        rec = {
            "system": sys,
            "input": str(r["Content"]),
            "service": str(r.get("Component", "")),
            "event_type": f"{sys}:{r['EventId']}",
            "severity": SEV_MAP.get(raw_level, raw_level),
            "identifiers": ids,
            "recommended_action": None,          # teacher fills this
            "_event_template": str(r["EventTemplate"]),   # traceability
            "_raw_level": raw_level,
        }
        if sys == "BGL":
            rec["_bgl_label"] = str(r["Label"])  # real anomaly ground truth
        recs.append(rec)
    return recs, fails

def main():
    os.makedirs(OUT, exist_ok=True)
    all_recs = []
    for sys in SYSTEMS:
        recs, fails = build_system(sys)
        print(f"{sys:10s} {len(recs)} records, {fails} alignment failures")
        all_recs.extend(recs)
    with open(os.path.join(OUT, "records_partial.jsonl"), "w") as f:
        for r in all_recs:
            f.write(json.dumps(r) + "\n")
    print(f"TOTAL {len(all_recs)} records -> data/labeled/records_partial.jsonl")

if __name__ == "__main__":
    main()
