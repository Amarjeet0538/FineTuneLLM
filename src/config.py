"""
config.py - single source of truth for Week 2 onwards.

Every script imports from here, so the split settings, the label vocabulary
and the PROMPT FORMAT can never drift apart between the baseline, the
fine-tune and the five quantisation levels.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------- files ----------
RECORDS     = ROOT / "data" / "labeled" / "records_labeled.jsonl"   # Week 1 answer key (input)
SPLIT_DIR   = ROOT / "data" / "split"
STAGING     = SPLIT_DIR / "staging.jsonl"        # every record + which bucket it landed in
REVIEW_CSV  = SPLIT_DIR / "eval_review.csv"      # you hand-check this (one row per template+severity)
TRAIN       = SPLIT_DIR / "train.jsonl"
DEV         = SPLIT_DIR / "dev.jsonl"            # for choosing checkpoints/settings in Week 3
MANIFEST    = SPLIT_DIR / "split_manifest.json"
EVAL_FROZEN = ROOT / "data" / "eval_frozen.jsonl"  # never edited after freeze_eval.py runs
EVAL_HASH   = ROOT / "data" / "eval_frozen.sha256"
RESULTS     = ROOT / "results"

# ---------- split settings ----------
SEED = 42
# A "template" = one LogHub EventId (our event_type field, e.g. "HDFS:E5").
# The split is done by TEMPLATE, never by line, so an unseen-template eval
# line can never have a twin in the training data.
EVAL_UNSEEN_FRACTION = 0.15   # share of templates (per action class) held out for eval
DEV_UNSEEN_FRACTION  = 0.10   # share of templates held out for dev
UNSEEN_LINES_CAP     = 12     # max distinct lines taken from each held-out template
SEEN_LINES_PER_TEMPLATE = 3   # distinct lines per training template for the "seen" eval part

# ---------- label vocabulary (decided BEFORE freezing, never after) ----------
# Classes with too few templates to both train and evaluate on are merged.
ACTION_MERGE = {
    "check_disk": "investigate",
    "check_auth": "investigate",
    "restart_service": "investigate",
}
ACTIONS = ["no_action", "investigate", "escalate_oncall", "check_network"]
MIN_TEMPLATES_PER_ACTION = 5
SEVERITIES = ["INFO", "WARNING", "ERROR", "FATAL"]

# What the model must output (event_type is NOT output: it is an opaque LogHub
# ID like "BGL:E49" that no model can infer from the text; kept as metadata).
OUTPUT_KEYS = ["service", "severity", "identifiers", "recommended_action"]

# ---------- the prompt (frozen together with the eval set) ----------
INSTRUCTIONS = (
    "You are a log triage tool. The user message is one log message from a "
    "production system (HDFS, BGL, OpenStack or Spark). Reply with ONLY one JSON "
    "object and nothing else, with exactly these four keys:\n"
    '"service": the name of the software component that wrote the message '
    "(for example dfs.DataNode or KERNEL).\n"
    '"severity": one of ' + ", ".join(SEVERITIES) + ".\n"
    '"identifiers": a list of every variable value in the message (IDs, IP '
    "addresses, paths, numbers), copied exactly, in the order they appear. "
    "Use [] if there are none.\n"
    '"recommended_action": one of ' + ", ".join(ACTIONS) + "."
)


def build_prompt(log_message: str) -> str:
    """SmolLM3 chat format in no-think mode, written out by hand.

    This is exactly what the official SmolLM3 chat template (Unsloth/llama.cpp
    copy) produces for [system, user] with reasoning off. Writing it by hand
    means Ollama's template guessing can't change what the model sees, and
    Week 3 training will reuse this same function.
    """
    return (
        "<|im_start|>system\n"
        "## Metadata\n\n"
        "Knowledge Cutoff Date: June 2025\n"
        "Reasoning Mode: /no_think\n\n"
        "## Custom Instructions\n\n"
        + INSTRUCTIONS + "\n\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        + log_message + "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n"
    )


def format_target(rec: dict) -> str:
    """The exact answer text the model is trained to produce (Week 3)."""
    return json.dumps({k: rec[k] for k in OUTPUT_KEYS}, ensure_ascii=False)


STOP = ["<|im_end|>"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_frozen():
    """Refuse to go on if the eval set is missing or was changed after freezing."""
    if not EVAL_FROZEN.exists() or not EVAL_HASH.exists():
        sys.exit("STOPPED: eval set not frozen yet - run freeze_eval.py first.")
    expected = EVAL_HASH.read_text(encoding="utf-8").split()[0]
    if sha256_of(EVAL_FROZEN) != expected:
        sys.exit("STOPPED: eval_frozen.jsonl does not match its saved hash - "
                 "it was modified after freezing. Restore it from git.")
