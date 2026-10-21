"""
run_ollama.py - get a model's raw answers for every eval (or dev) example.

    python src\\run_ollama.py --model hf.co/ggml-org/SmolLM3-3B-GGUF:Q8_0 --tag base_q8 --split dev --limit 5
    python src\\run_ollama.py --model hf.co/ggml-org/SmolLM3-3B-GGUF:Q8_0 --tag base_q8

Fixed for EVERY model you ever score (baseline, fine-tune, all 5 quant levels):
  raw prompt from config.build_prompt, greedy decoding (temperature 0), seed,
  2048-token context, 384-token output limit, no JSON-forcing ("format" is NOT
  used - whether the model produces valid JSON on its own is part of the score).

Safe to stop with Ctrl+C and re-run: finished examples are skipped.
"""
import argparse, json, sys, time, urllib.error, urllib.request
import config as C

OPTIONS = {"temperature": 0, "seed": C.SEED, "num_ctx": 2048, "num_predict": 384,
           "stop": C.STOP}


def http(url, payload=None, timeout=900):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="exact Ollama model name")
    ap.add_argument("--tag", required=True, help="short name for the output file, e.g. base_q8")
    ap.add_argument("--split", choices=["eval", "dev"], default="eval")
    ap.add_argument("--limit", type=int, default=0, help="only the first N examples (smoke test)")
    ap.add_argument("--url", default="http://localhost:11434")
    args = ap.parse_args()

    if args.split == "eval":
        C.verify_frozen()
        data = C.read_jsonl(C.EVAL_FROZEN)
    else:
        data = C.read_jsonl(C.DEV)
    if args.limit:
        data = data[:args.limit]

    # is Ollama up, and is the model pulled?
    try:
        tags = http(args.url + "/api/tags", timeout=10)
    except (urllib.error.URLError, ConnectionError):
        sys.exit("STOPPED: can't reach Ollama at " + args.url + " - start the Ollama app first.")
    models = {m["name"].lower(): m for m in tags.get("models", [])}
    info = models.get(args.model.lower()) or models.get(args.model.lower() + ":latest")
    if not info:
        sys.exit(f"STOPPED: model '{args.model}' not found. Run:  ollama pull {args.model}\n"
                 f"Models you have: {sorted(models) or 'none'}")
    quant = info.get("details", {}).get("quantization_level", "?")

    C.RESULTS.mkdir(exist_ok=True)
    out = C.RESULTS / f"preds_{args.split}_{args.tag}.jsonl"
    done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    todo = [r for r in data if r["id"] not in done]
    print(f"model {args.model} (quant {quant}) | {args.split}: {len(data)} examples, "
          f"{len(done)} already done, {len(todo)} to go -> {out}")

    t0 = time.time()
    with open(out, "a", encoding="utf-8", newline="\n") as f:
        for i, r in enumerate(todo, 1):
            try:
                res = http(args.url + "/api/generate", {
                    "model": args.model, "prompt": C.build_prompt(r["input"]),
                    "raw": True, "stream": False, "keep_alive": "15m", "options": OPTIONS})
            except urllib.error.HTTPError as e:
                sys.exit(f"STOPPED: Ollama error {e.code}: {e.read().decode('utf-8', 'replace')}")
            f.write(json.dumps({
                "id": r["id"], "model": args.model, "quant": quant,
                "raw_output": res.get("response", ""),
                "done_reason": res.get("done_reason"),
                "prompt_tokens": res.get("prompt_eval_count"),
                "output_tokens": res.get("eval_count"),
                "seconds": round(res.get("total_duration", 0) / 1e9, 3),
            }, ensure_ascii=False) + "\n")
            f.flush()
            if i == 1 or i % 10 == 0 or i == len(todo):
                per = (time.time() - t0) / i
                print(f"  {i}/{len(todo)}  {per:.1f}s per example, ~{per * (len(todo) - i) / 60:.0f} min left")
    print("done.")


if __name__ == "__main__":
    main()
