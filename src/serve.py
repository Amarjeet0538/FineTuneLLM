"""
serve.py - the log-triage API (Week 5).

Start it (Ollama must be running):
    uvicorn serve:app --app-dir src --port 8000

Then:
    GET  /                                          ->  the triage page: paste a log line, see the record
    POST /triage   {"log": "<one raw log line>"}  ->  the structured record
    GET  /health                                    ->  is Ollama up, is the model there?
    GET  /docs                                      ->  the raw API test page

It sends EXACTLY what the scoring runs sent: the same prompt (config.build_prompt),
the same settings (run_ollama.OPTIONS), and it reads the answer with the same
parser (score.parse). So the API answers the way the scored model answered.

Optional settings (environment variables):
    OLLAMA_URL   where Ollama is      default http://127.0.0.1:11434
    MODEL        which Ollama model   default smollm3-ft-q5km  (the Week 4 sweet spot)
"""
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import config as C
from run_ollama import OPTIONS
from score import CAT_FIELDS, parse

# 127.0.0.1, NOT localhost: on Windows "localhost" is tried as IPv6 (::1) first, Ollama only listens
# on 127.0.0.1, and Windows retries the refused connection for ~2 s per request (found in Week 6).
# Docker is not affected: docker-compose.yml sets OLLAMA_URL=http://ollama:11434.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
MODEL = os.environ.get("MODEL", "smollm3-ft-q5km")
MAX_LOG_CHARS = 2000   # the prompt must fit in the 2048-token context with room for the answer
UI_FILE = Path(__file__).with_name("ui.html")   # the page; it only calls /triage and /health

app = FastAPI(title="Log-Triage SLM", version="1.0",
              description="Turns one raw log line into a structured triage record.")


class TriageRequest(BaseModel):
    log: str = Field(..., min_length=1, max_length=MAX_LOG_CHARS,
                     description="One raw log message (text only, no level/component header)")


def schema_ok(o) -> bool:
    """Same rule as score.py: all 4 keys, strings where strings are expected,
    identifiers is a list of strings."""
    return (isinstance(o, dict)
            and all(k in o for k in C.OUTPUT_KEYS)
            and all(isinstance(o[k], str) for k in CAT_FIELDS)
            and isinstance(o["identifiers"], list)
            and all(isinstance(x, str) for x in o["identifiers"]))


def ms(ns):
    return round(ns / 1e6, 1) if isinstance(ns, (int, float)) else None


@app.get("/", include_in_schema=False)
async def home():
    """The triage page (src/ui.html). It is read on every request, so it never goes stale."""
    if not UI_FILE.is_file():
        raise HTTPException(500, "ui.html is missing: it must be in the same folder as serve.py.")
    return HTMLResponse(UI_FILE.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            tags = (await client.get(OLLAMA_URL + "/api/tags")).json()
            version = (await client.get(OLLAMA_URL + "/api/version")).json().get("version")
    except httpx.HTTPError:
        raise HTTPException(503, f"Can't reach Ollama at {OLLAMA_URL}. Is the Ollama app running?")
    names = {m.get("name", "").lower() for m in tags.get("models", [])}
    if MODEL.lower() not in names and MODEL.lower() + ":latest" not in names:
        raise HTTPException(503, f"Ollama is up but model '{MODEL}' is not in it. "
                                 f"Models found: {sorted(names)}")
    return {"status": "ok", "ollama_version": version, "model": MODEL}


@app.post("/triage")
async def triage(req: TriageRequest):
    text = req.log.strip()
    if not text:
        raise HTTPException(422, "log is empty")

    payload = {"model": MODEL, "prompt": C.build_prompt(text), "raw": True,
               "stream": False, "keep_alive": "15m", "options": OPTIONS}
    try:
        async with httpx.AsyncClient(timeout=600) as client:
            r = await client.post(OLLAMA_URL + "/api/generate", json=payload)
    except httpx.TimeoutException:
        raise HTTPException(504, "Ollama took longer than 10 minutes to answer.")
    except httpx.HTTPError:
        raise HTTPException(503, f"Can't reach Ollama at {OLLAMA_URL}. Is the Ollama app running?")
    if r.status_code != 200:
        raise HTTPException(502, f"Ollama error {r.status_code}: {r.text[:300]}")

    res = r.json()
    raw = res.get("response", "")
    _, obj = parse(raw)
    valid = schema_ok(obj)
    allowed = (valid
               and obj["severity"].strip().upper() in C.SEVERITIES
               and obj["recommended_action"].strip().lower() in C.ACTIONS)
    return {
        "result": {k: obj[k] for k in C.OUTPUT_KEYS} if valid else None,
        "schema_valid": valid,          # False = the model's answer couldn't be read as the 4-key record
        "allowed_values": allowed,      # severity and action are from the fixed lists
        "raw_output": raw,              # exactly what the model wrote
        "truncated": res.get("done_reason") == "length",
        "model": MODEL,
        "tokens": {"prompt": res.get("prompt_eval_count"), "output": res.get("eval_count")},
        "timing_ms": {"total": ms(res.get("total_duration")),
                      "load": ms(res.get("load_duration")),
                      "prompt": ms(res.get("prompt_eval_duration")),
                      "generate": ms(res.get("eval_duration"))},
    }
