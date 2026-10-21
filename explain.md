# Log-Triage SLM: the whole project, explained from zero

This guide is for interview prep. It assumes you know nothing about language models. It starts with the
basic ideas, then goes through every stage of the project: what was built, how, which tools were used,
and why each decision was made. Every number here comes from the files in this repo
(`results/`, `data/split/split_manifest.json`, `data/week3/build_report.json`, `results/run_info.json`,
`results/week4_quant_info.json`).

---

## Table of contents

1. [The project in one paragraph](#1-the-project-in-one-paragraph)
2. [Background you need first (the concepts)](#2-background-you-need-first)
3. [The problem and the objectives](#3-the-problem-and-the-objectives)
4. [Tech stack and tools](#4-tech-stack-and-tools)
5. [Architecture: the big picture](#5-architecture-the-big-picture)
6. [Stage by stage: the full flow](#6-stage-by-stage-the-full-flow)
7. [Optimisation in depth: what, why and how](#7-optimisation-in-depth)
8. [Evaluation methodology: how we know the numbers are honest](#8-evaluation-methodology)
9. [Results and what they mean](#9-results-and-what-they-mean)
10. [Engineering practices worth mentioning](#10-engineering-practices-worth-mentioning)
11. [Limitations (say these before the interviewer does)](#11-limitations)
12. [Interview questions and model answers](#12-interview-questions-and-model-answers)
13. [Glossary](#13-glossary)

---

## 1. The project in one paragraph

Servers produce millions of log lines a day, such as
`Received block blk_-1608999687919862906 of size 91178 from /10.250.10.6`.
An on-call engineer has to read them and decide what matters. This project trains a **small
language model** (SmolLM3, 3 billion parameters) to read **one raw log line** and return a **structured JSON
record**: which software component wrote it, how severe it is, which IDs and values it contains, and what a
human should do about it. The model is then **compressed (quantised)** to five precision levels, and every
level is measured for **accuracy** (on a frozen test set) and **cost** (memory, latency, throughput on an
ordinary laptop CPU). The real question is **"how much accuracy survives compression, and what does
each compression level buy you?"** The answer: Q5_K_M (about 5.8 bits per weight) is as accurate as full
16-bit precision while being **64% smaller** and **2.2x faster per answer**.

Example:

```
Input : BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.74.79:50010 is added to blk_-2827716238972737794 size 67108864
Output: {"service": "dfs.DataNode$DataXceiver", "severity": "INFO",
         "identifiers": ["-2827716238972737794"], "recommended_action": "no_action"}
```

---

## 2. Background you need first

Learn these concepts first. Everything later builds on them.

### 2.1 What is a language model (LLM / SLM)?

A language model is a neural network that has learned to **predict the next piece of text**. You give it
text, it outputs a probability for every possible next token, one is picked, it is appended, and the
loop repeats. That loop is how it "writes".

- **LLM** = large language model (tens to hundreds of billions of parameters, e.g. GPT-4-class).
- **SLM** = small language model (roughly 0.5B to 8B parameters). It runs on a laptop and is cheap to fine-tune.
- **Parameters / weights**: the numbers inside the network that were learned during training. SmolLM3-3B has
  about **3.08 billion** of them (`3,075,098,624`, from `week4_quant_info.json`).

### 2.2 Tokens

Models don't read characters or words. They read **tokens**, which are chunks of text (often parts of words).
`"blockMap updated"` might become 3 or 4 tokens. Everything is counted in tokens: the context window (how much
the model can see at once, 2048 here), the output limit (384 here) and speed (tokens per second).

### 2.3 Prompt, chat template, special tokens

Chat models are trained on text in a fixed format with **special tokens** that mark who is speaking:

```
<|im_start|>system
...instructions...<|im_end|>
<|im_start|>user
...the log line...<|im_end|>
<|im_start|>assistant
```

The model then writes its answer and finishes with `<|im_end|>` (the **stop token**). If the prompt format
is even slightly different from what the model was trained on, quality drops. That is why this project
writes the template **by hand** (`config.build_prompt`).

### 2.4 Pre-training vs fine-tuning

- **Pre-training**: the model learns general language from trillions of tokens. Someone else already did this
  (Hugging Face trained SmolLM3).
- **Fine-tuning**: you continue training on a **small, task-specific dataset** (here, 1,057 examples) so the
  model learns *your* task and *your* output format.
- **Supervised fine-tuning (SFT)**: the data is (prompt, correct answer) pairs, and the model is trained to
  produce the answer.

### 2.5 LoRA and QLoRA (how fine-tuning was made cheap)

Updating all 3 billion weights needs a lot of GPU memory, because every weight also needs its gradients
and optimiser state. **LoRA (Low-Rank Adaptation)** avoids that:

- The original weights are **frozen** and never change.
- Next to each big weight matrix `W` (shape d x d), two small matrices `A` (d x r) and `B` (r x d) are added, where
  `r` (the **rank**) is tiny, here **16**. The layer computes `W·x + (B·A)·x · (alpha / r)`.
- Only `A` and `B` are trained, which is usually well under 1% of the parameters. The result is a small file called an
  **adapter**.

**QLoRA** = LoRA + store the frozen base model in **4-bit** during training. The base takes about a quarter
of the memory, so a 3B model fits comfortably on a free 16 GB T4 GPU.

### 2.6 Quantisation (the core of this project)

A weight is normally stored as a 16-bit floating-point number (2 bytes). **Quantisation stores weights with
fewer bits**, for example 8, 5, 4 or 3. Fewer bits means:

- **A smaller file and less RAM.** 3.08B weights x 2 bytes = 6.16 GB at F16, but about 2.2 GB at Q5_K_M.
- **Faster generation on a CPU.** Producing each token means reading *every* weight from RAM once, so less data
  to read means faster tokens (see section 7.3.2).
- **Some loss of precision.** Rounding to fewer levels adds error, and the open question is how much of
  that error shows up as wrong answers. Measuring that is the point of the project.

How it works, in simple terms: weights are split into **blocks** (e.g. 32 weights). For each block a **scale**
is stored (and sometimes a minimum), and each weight becomes a small integer. To use the weight you compute
`weight ≈ scale x integer (+ min)`.

### 2.7 GGUF, llama.cpp, Ollama

- **llama.cpp**: a C/C++ program that runs language models fast on ordinary CPUs (and GPUs). It defines the
  quantisation formats used here.
- **GGUF**: llama.cpp's file format. One file holds the weights (in any quantisation), the tokenizer and all
  model settings.
- **Ollama**: a friendly wrapper around llama.cpp. It runs a local server on port 11434 with a simple HTTP API
  (`/api/generate`), manages models, and loads and unloads them from memory.
- **Modelfile**: Ollama's recipe file for importing a GGUF (`FROM ./smollm3-logtriage-q5_k_m.gguf`).

### 2.8 Teacher-student distillation

When you lack labels for something, you can ask a **bigger, smarter model (the teacher)** to label the
data, then train the **small model (the student)** on those labels. Here, only the `recommended_action`
field was labelled by a teacher (via the Groq or Gemini APIs). The other fields come from real human-made
annotations.

### 2.9 Accuracy, precision, recall, F1, macro-F1

- **Accuracy**: the fraction of answers that are right.
- **Precision** (for one class): of all the times the model said "escalate_oncall", how often was it right?
- **Recall** (for one class): of all the real "escalate_oncall" lines, how many did the model find?
- **F1**: the harmonic mean of precision and recall. It is high only when both are high.
- **Macro-F1**: compute F1 for each class separately, then **average them equally**. A rare class counts as much
  as a common one, and that matters here (section 8.4).

---

## 3. The problem and the objectives

### 3.1 The problem

Log triage is repetitive and time-critical. Rule-based systems (regexes) break when the log format changes.
Big cloud LLMs work but are expensive, slow, and send your logs to a third party. A **small model running
locally** would be cheap and private. Small models are weaker, though, and squeezing them onto cheap hardware
(quantisation) makes them weaker still. **How much weaker, exactly?**

### 3.2 Objectives

1. **Build a clean dataset**: log line in, structured record out, with ground truth wherever possible.
2. **Build a test set that cannot be gamed**: split by log *pattern*, frozen and hashed before training.
3. **Measure a baseline**: the untuned model, plus a "always answer the most common action" baseline.
4. **Fine-tune** SmolLM3-3B with QLoRA on free hardware (Kaggle T4).
5. **Quantise** to F16, Q8_0, Q5_K_M, Q4_K_M and Q3_K_M, and **score every level on the same frozen set**.
6. **Benchmark the cost** of each level (RAM, time to first token, total latency, throughput) on a laptop CPU.
7. **Load-test** the best candidates with several concurrent requests.
8. **Serve** it as a real API (FastAPI + web page) and package it with Docker.
9. **Recommend** a level, backed by numbers.

### 3.3 The output schema

| Key | Type | Where the label comes from |
|---|---|---|
| `service` | string | LogHub's `Component` column (ground truth) |
| `severity` | one of `INFO`, `WARNING`, `ERROR`, `FATAL` | LogHub's `Level` column, normalised (ground truth) |
| `identifiers` | list of strings | Extracted by matching the line against LogHub's verified template (ground truth) |
| `recommended_action` | one of `no_action`, `investigate`, `escalate_oncall`, `check_network` | **Teacher model**, one label per (pattern, severity) group, reviewed by a human |

Important: the model sees **only the message text**. The log header (which contains the level and the component) is
stripped, so the model can't just copy the answer from the input. This is why `service` is hard (section 11).

---

## 4. Tech stack and tools

| Layer | Tool (pinned version) | Why this one |
|---|---|---|
| Language | Python (3.13 in Docker) | ML ecosystem standard |
| Data handling | pandas 3.0.2 | Reading the LogHub CSVs |
| Dataset | **LogHub-2k** (HDFS, BGL, OpenStack, Spark): 8,000 lines, 213 patterns | Public log benchmark with **hand-corrected templates**, so it provides free ground truth |
| Teacher labelling | groq 1.7.0, google-genai 2.23.0 (+ a `mock` provider) | Free-tier LLM APIs for the one field without ground truth |
| Base model | **SmolLM3-3B** (`HuggingFaceTB/SmolLM3-3B`) | Strong 3B open model, Apache-2.0, has a "no-think" mode, small enough for a laptop |
| Training | PyTorch 2.10 (CUDA 12.8), transformers 5.17.0, **peft 0.21.0** (LoRA), **bitsandbytes 0.50.2** (4-bit), accelerate 1.15.0 | Standard Hugging Face QLoRA stack |
| Training hardware | **Kaggle free GPU: one NVIDIA Tesla T4** (16 GB) | Free. About 30 minutes for the whole run |
| Conversion / quantisation | **llama.cpp**, pinned to commit `b23efaa2…`: `convert_hf_to_gguf.py`, `llama-quantize` | The standard for CPU inference. Pinned so results can be reproduced |
| Inference runtime | **Ollama 0.34.2** | Simple local server. **The same runtime for accuracy and cost**, so the two axes agree |
| API | **FastAPI 0.143.0**, uvicorn 0.54.0, httpx 0.28.1 (async), Pydantic | Async, auto-generated `/docs`, input validation |
| Front end | One static `ui.html` | A small page that only calls the API |
| Containers | Docker + docker compose (`ollama/ollama:0.34.2` + the API) | One-command demo |
| Benchmark tooling | psutil 7.2.2 (process memory), Windows `ctypes` (power mode, sleep prevention) | Reads peak RAM of the `llama-server` process |
| Charts | matplotlib 3.11.2 | Trade-off and load figures |
| Model hosting | Hugging Face Hub (`huggingface_hub` 1.33.0) | Public weights: [shubham-alyan24/SmolLM3-3B-logtriage-GGUF](https://huggingface.co/shubham-alyan24/SmolLM3-3B-logtriage-GGUF) |
| Benchmark machine | Laptop, Intel **Core i5-13500H, CPU only**, Windows | "Ordinary hardware" is part of the question |

---

## 5. Architecture: the big picture

There are two architectures to understand: the **offline ML pipeline** (how the model was made and
measured) and the **online serving architecture** (how a request is answered).

### 5.1 Offline pipeline (data → model → measurements)

```
 LogHub-2k CSVs (data/raw/)
        │  build_dataset.py        ground truth: service, severity, identifiers
        ▼
 records_partial.jsonl
        │  label_with_teacher.py   teacher LLM labels recommended_action per (pattern, severity) group
        │  cleanup_bgl_dumps.py    one-off fix for BGL register-dump lines
        ▼
 records_labeled.jsonl  (8,000 lines)
        │  split_data.py           split BY PATTERN → train / dev / eval  +  eval_review.csv
        │  (human review of the CSV)
        │  freeze_eval.py          eval_frozen.jsonl + its SHA-256 hash  ← never changes again
        ▼
 train.jsonl / dev.jsonl / eval_frozen.jsonl
        │  build_train_set.py      dedupe, cap no_action groups, turn into (prompt, answer) pairs
        ▼
 train_sft.jsonl (1,057)  dev_sft.jsonl (145)
        │  ── uploaded to Kaggle ──
        │  week3_finetune_smollm3.ipynb   QLoRA on T4 → merge adapter → GGUF Q8_0
        │  week4_quantise_smollm3.ipynb   merged → F16 GGUF → llama-quantize → Q5_K_M, Q4_K_M, Q3_K_M
        │  week4_publish_hf.ipynb         upload all 5 + adapter to Hugging Face (hash-checked)
        ▼
 5 GGUF files  ──ollama create──►  Ollama models smollm3-ft-{f16,q8,q5km,q4km,q3km}
        │
        ├─ run_ollama.py + score.py      → ACCURACY  (results/preds_*.jsonl, scores_*.json)
        ├─ bench.py                      → COST, one request at a time (results/bench_*.json)
        ├─ loadtest.py                   → COST under concurrency (results/load_*.json)
        └─ week6_table / week7_figure / week7_load_table  → tables + figures for the report
```

### 5.2 Online serving architecture (one request)

```
 Browser (ui.html)  or  any HTTP client
        │  POST /triage {"log": "..."}
        ▼
 ┌──────────────────────── FastAPI app (src/serve.py) ────────────────────────┐
 │ 1. Pydantic validates: 1–2000 chars                                         │
 │ 2. prompt = config.build_prompt(log)          ← SAME function as scoring     │
 │ 3. POST Ollama /api/generate  raw=True, options = run_ollama.OPTIONS         │
 │    (temperature 0, seed 42, num_ctx 2048, num_predict 384, stop <|im_end|>)  │
 │ 4. score.parse(raw)                            ← SAME parser as scoring      │
 │ 5. schema check + allowed-values check                                       │
 │ 6. return result + raw_output + tokens + timings                             │
 └──────────────────────────────────────────────────────────────────────────────┘
        │ HTTP (127.0.0.1:11434, or ollama:11434 in Docker)
        ▼
 Ollama 0.34.2  →  llama-server process  →  GGUF model in RAM (default smollm3-ft-q5km)
```

The key design idea: **"what you measured is what you serve."** The API imports the prompt builder, the
generation settings and the parser from the very modules used for scoring. It does not copy them. The served
model therefore behaves exactly like the scored one, and the benchmark found all 5 x 328 answers
**character-for-character identical** to the scoring run.

### 5.3 Docker deployment

`docker-compose.yml` defines two services:

- `ollama`: the official `ollama/ollama:0.34.2` image (same version as scoring). It mounts `./models` read-only,
  and a named volume keeps imported models between restarts.
- `api`: built from the `Dockerfile` (`python:3.13-slim`). It copies **only the 5 files** the API needs, runs as
  a **non-root user**, and is exposed on host port **8080** so it doesn't clash with the native API on 8000.
  `OLLAMA_URL=http://ollama:11434` uses Docker's internal DNS.

### 5.4 Source map

| File | Role |
|---|---|
| `src/config.py` | **Single source of truth**: paths, seed, label vocabulary, the frozen prompt, the answer format, the hash check |
| `src/build_dataset.py` | Parses LogHub, extracts the ground-truth fields |
| `src/label_with_teacher.py` | Teacher distillation of `recommended_action` |
| `src/cleanup_bgl_dumps.py` | One-off fix for BGL crash-dump lines |
| `src/split_data.py`, `src/freeze_eval.py` | Pattern-based split, human review, freezing |
| `src/build_train_set.py` | Builds the SFT pairs (dedupe + cap) |
| `notebooks/*.ipynb` | Fine-tuning, quantisation and publishing (run on Kaggle) |
| `src/run_ollama.py`, `src/score.py` | Inference over a dataset, then scoring |
| `src/serve.py`, `src/ui.html` | The API and the web page |
| `src/bench.py`, `src/loadtest.py` | Cost benchmarks |
| `src/week6_table.py`, `src/week7_figure.py`, `src/week7_load_table.py` | Tables and figures |

---

## 6. Stage by stage: the full flow

### Stage 1: building the dataset (`build_dataset.py`)

**Input**: LogHub-2k "structured" CSVs. Each row has `Content` (the message), `Level`, `Component`,
`EventId` (pattern ID) and `EventTemplate` (the pattern with `<*>` in place of the variable parts). These
templates were **corrected by hand** by LogHub's authors, which makes them reliable ground truth.

**How each field is built**:

- `service` ← `Component`.
- `severity` ← `Level`, normalised (`WARN`→`WARNING`, `SEVERE`→`ERROR`, etc.).
- `event_type` ← `"<system>:<EventId>"`, e.g. `HDFS:E5`. This is metadata used for splitting. **It is not output**
  because it's an arbitrary ID that no model could infer.
- `identifiers` ← the template is turned into a regex: every `<*>` becomes `(.*?)` and the rest is escaped.
  It is matched against the message, and the captured groups are the identifiers.

**Design decision**: **Drain3** (a popular automatic log parser) was deliberately **not** used. On these 2k
sets it matched the corrected ground-truth template 0% of the time and would have added parser errors
to the labels.

### Stage 2: teacher labelling (`label_with_teacher.py`)

Only `recommended_action` needs a judgement call ("should someone act on this?"), so only that field comes from a teacher LLM.

Key design choices:

- **Per-group labelling.** The right action depends on the *pattern and severity*, not on the specific IDs in
  the line. So instead of 8,000 API calls, there is **one call per (event_type, severity) group** (about 222 groups),
  and the label is copied to every line in the group. That is about **36x fewer calls**, and the labels are consistent (two lines of the
  same pattern can never get different actions). A human only has to review about 222 decisions instead of 8,000.
- **Closed vocabulary.** The teacher must choose from a fixed list. Near-misses are repaired with a synonym table
  (`"escalate"`→`escalate_oncall`). Anything else becomes `NEEDS_REVIEW` and is **never guessed**.
- **Resumable.** Each decision is appended to `action_cache.jsonl` straight away, so a crash or rate limit
  loses nothing.
- **Retries with exponential backoff** (1 s, 2 s, then mark for review).
- **A `mock` provider** (simple keyword rules) tests the plumbing without spending API calls. It is not used for training.
- Teacher calls use **temperature 0** for determinism.

`cleanup_bgl_dumps.py`: when a BGL machine crashes it dumps its CPU registers line by line, and every one
of those lines inherits the crash's `FATAL` level. The teacher had labelled some of them `escalate_oncall`. They are
fragments of a dump, not events you can act on, so that family was relabelled `no_action`. The real crash
events were left untouched.

### Stage 3: splitting and freezing (`split_data.py` → human review → `freeze_eval.py`)

This is the most important stage for **trustworthy results**.

**Label vocabulary decided first.** The teacher's original 7 actions were merged to 4, because
`check_disk`, `check_auth` and `restart_service` had too few patterns (fewer than 5) to both train
*and* evaluate on. They became `investigate` (`ACTION_MERGE` in `config.py`). This was decided **before** freezing.

**Split by pattern, not by line.** LogHub repeats lines heavily: one pattern has over 900 lines. A random line
split would put near-identical twins in train and test, so the model would just memorise and the
score would be inflated. Instead:

- **Unseen-pattern eval**: 15% of patterns (stratified per action class, so rare classes are represented) are
  held out **entirely**. The model never sees any line of these patterns. Up to 12 distinct lines are taken from each.
  **This is the honest test** of generalisation.
- **Seen-pattern eval**: 3 new distinct lines from patterns that *are* in training. This is the in-distribution test.
- **Dev**: another 10% of patterns held out, plus 1 line from some training patterns. It is used only to choose the
  checkpoint.
- **Work on distinct texts**: identical copies of a held-out line are discarded, so no eval line has a twin in train.
- **Hard guarantees with `assert`**: the script fails loudly if any eval or dev text is in train, or if any
  unseen pattern appears in train.

**Human review.** `eval_review.csv` has one row per (pattern, severity) group in eval. A human checks each
teacher label. One correction was made (`BGL:E80|FATAL`: `investigate` → `escalate_oncall`), and it is
applied **everywhere** (train, dev and eval) so the training labels agree with the answer key.

**Freezing.** `freeze_eval.py` writes `data/eval_frozen.jsonl` and its **SHA-256 hash**. From then on:

- `config.verify_frozen()` runs before any scoring and **refuses to continue** if the file's hash changed.
- `split_data.py` and `freeze_eval.py` refuse to run again.
- `.gitattributes` sets `*.jsonl -text`, so Git never rewrites line endings (CRLF/LF), which would change the hash.

Final counts: **train 5,451 lines, dev 145, eval 329** (185 seen-pattern + 144 unseen-pattern).
Eval by action: no_action 282, investigate 25, check_network 16, escalate_oncall 6.

### Stage 4: building the training set (`build_train_set.py`)

1. **Leak check**: no train or dev text may be in the frozen eval set.
2. **Remove exact duplicates** (same text + same answer): 5,451 → 4,500 distinct pairs.
3. **Cap each `no_action` (pattern, severity) group at 20 examples** (picked at random with the seed). Rare classes are
   **never** capped. Result: **1,057 training examples**:
   no_action 910, check_network 84, investigate 52, escalate_oncall 11.
4. Each example becomes `prompt = config.build_prompt(line)` and `answer = config.format_target(record)`
   (compact JSON with the 4 keys in a fixed order).

Why the cap? See section 7.1.1.

### Stage 5: fine-tuning (`notebooks/week3_finetune_smollm3.ipynb`, Kaggle T4)

Settings (fixed **before** training, taken from common QLoRA defaults and **not** tuned on eval):

| Setting | Value |
|---|---|
| Base model loaded in | 4-bit **NF4**, double quantisation, fp16 compute |
| LoRA | rank **16**, alpha **32**, dropout 0.05, on **all linear layers** |
| Learning rate | **2e-4**, cosine schedule, 5% warmup |
| Batch | 4 per step x 4 gradient-accumulation steps = **effective batch 16** |
| Epochs | 3 (dev loss checked after each) |
| Max length | 1,024 tokens (0 train and 1 dev example dropped) |
| Precision | fp16 mixed precision, gradient checkpointing |
| Optimiser | AdamW, no weight decay |
| Loss | **answer tokens only** (prompt masked) |
| Seed | 42 |

The process, step by step:

1. **Smoke test first.** `SMOKE_TEST=True` runs 10 steps on 64 examples, then merges and converts, which tests the whole
   pipeline end to end in about 20 minutes before the real run is committed.
2. **BOS check.** Does llama.cpp/Ollama add a beginning-of-sequence token in front of the prompt? The notebook asks the
   *same pinned converter* and finds it does **not** for this tokenizer (`smaug-bpe` pre-tokenizer), so
   training doesn't either. Training input must match inference input token for token.
3. **Token cross-check.** It records the prompt-token count of one dev example (195) so it can later be compared with
   what Ollama reports.
4. **Loss masking.** Prompt tokens get the label `-100` (ignored by the loss). Only the answer +
   `<|im_end|>` is graded. An assert checks that the graded part decodes to exactly the answer.
5. **Train**, with a NaN watchdog callback that stops training if the loss becomes NaN or infinite.
6. **Checkpoint selection by dev loss only**: epoch 1 = 0.0496, epoch 2 = 0.0557, epoch 3 = 0.0517. **Epoch 1
   won** (checkpoint-67). Later epochs started to overfit slightly. Training took **30.2 minutes**.
7. **Merge.** The adapter is applied to a **full-precision (fp32) copy** of the base model on CPU, then the
   result is cast to bf16 (SmolLM3's native format). It is **not** merged into the 4-bit model, because merging into rounded weights would bake in
   the rounding error.
8. The original tokenizer and config files are copied back, so the fine-tuned model differs **only in its weights**.
9. **Convert to GGUF Q8_0**, then a **metadata diff**: every setting in the new GGUF (architecture, RoPE, tokenizer,
   vocabulary, BOS flag) must equal the base model's. Only `general.*` (name and file type) may differ.

### Stage 6: quantisation sweep (`notebooks/week4_quantise_smollm3.ipynb`)

| Level | How it was made |
|---|---|
| F16 | `convert_hf_to_gguf.py --outtype f16` from the merged bf16 model |
| Q8_0 | Reused from week 3, **not rebuilt**. Its hash is checked against the copy that was scored |
| Q5_K_M, Q4_K_M, Q3_K_M | `llama-quantize` from the F16 GGUF, **no importance matrix** |

Safety checks along the way:

- **The input is verified**: the Q8_0's SHA-256 on Kaggle must match the one scored on the laptop, which proves the merged
  model is the right one.
- **bf16 → f16 conversion audit.** bf16 and f16 are both 16-bit, but f16 has a **smaller range** (max 65,504)
  and more precision. Every one of the 3.08B weights was checked: **0 overflowed to infinity** (the largest
  weight was 9.5), 4,156 tiny values were flushed to zero and 689,731 were rounded slightly. That is harmless, and it was
  measured rather than assumed.
- **No NaN or infinity** anywhere in the F16 file.
- **Same metadata and same tensors** (names and shapes) in every level.
- **Type mix recorded**: K-quant levels are *mixes* of formats (section 7.2.3).
- The whole notebook took 13.9 minutes on 4 CPU cores.

### Stage 7: publishing (`notebooks/week4_publish_hf.ipynb`)

The 5 GGUFs, the LoRA adapter and a model card go to Hugging Face in **one commit**. Every file's hash is checked
against `week4_quant_info.json` **before** upload, and every uploaded file's size and hash are checked against Hugging Face's
copy **after**. The repo was created **private** first and made public after checking.

### Stage 8: scoring (`run_ollama.py` → `score.py`)

`run_ollama.py` sends every eval line to Ollama with **fixed settings shared by every model ever scored**:

- **Raw mode** (`raw: true`): Ollama's own chat template is bypassed and the hand-written prompt goes in as-is.
  Ollama's template guessing can't change what the model sees.
- **Greedy decoding** (temperature 0) + seed 42: deterministic, so there is no randomness in the comparison.
- `num_ctx` 2048, `num_predict` 384, stop at `<|im_end|>`.
- **No JSON-forcing** (`format` is not used). Whether the model produces valid JSON *on its own* is part of
  what is measured.
- **Resumable**: finished IDs are skipped on a re-run.

`score.py` details are in section 8.

### Stage 9: serving (`serve.py`, `ui.html`, Docker)

- `POST /triage` → `{result, schema_valid, allowed_values, raw_output, truncated, model, tokens, timing_ms}`.
- `GET /health` → checks Ollama is reachable and the model is installed (503 if not).
- `GET /` → the web page. `GET /docs` → the auto-generated Swagger UI.
- **Clear error mapping**: 422 for bad input, 503 when Ollama is unreachable, 504 on timeout, 502 for an Ollama error.
- `keep_alive: 15m` keeps the model in RAM between requests, which avoids a multi-second reload.
- `MAX_LOG_CHARS = 2000`, so the prompt plus the answer fit within the 2,048-token context.
- `MODEL` and `OLLAMA_URL` are configured through environment variables.

### Stage 10: benchmarks (`bench.py`, `loadtest.py`)

Covered in section 7.4, because benchmarking *is* the measurement half of optimisation.

---

## 7. Optimisation in depth

"Optimisation" in this project means three separate things. Keep them apart in an interview:

1. **Training-time optimisation**: making fine-tuning fit on a free GPU and learn the right thing.
2. **Model compression (quantisation)**: making the trained model smaller and faster to run.
3. **Inference / serving optimisation**: making each request fast and the system efficient under load.

### 7.1 Training-time optimisation

#### 7.1.1 Data optimisation: deduplication and class-balance capping

**Problem**: 86% of eval lines (and over 94% of raw training lines) are `no_action`, and one pattern alone has 900+
lines. If you train on that, the model learns "always say no_action". That gives high accuracy and is useless,
because it never flags anything.

**What was done**:

- Exact duplicates removed (5,451 → 4,500). Repeating the same example adds no information, only more weight
  on it.
- Each `no_action` group capped at **20**. Rare classes are never capped.
- Result: 1,057 examples. no_action falls from 5,145 lines to 910, and every rare example is kept.

**Why it works**: the model sees the rare classes in a much larger share of steps, and training is
**5x faster** (1,057 vs 5,451 examples). You lose very little, because 900 copies of the same pattern teach
nothing new after the first few.

**Why not oversample the rare classes instead?** With only 11 distinct `escalate_oncall` lines, repeating them would
encourage memorisation. Undersampling the majority class is the safer choice.

#### 7.1.2 QLoRA: 4-bit NF4 base + LoRA adapters

**Problem**: full fine-tuning of 3B parameters in fp16 needs roughly 3B x (2 bytes weights + 2 gradients +
8 for Adam state) ≈ 36+ GB, which is far more than a 16 GB T4.

**How QLoRA solves it**:

- **The frozen base is stored in 4-bit NF4** (about 0.5 bytes per weight, so roughly 1.5–2 GB). *NF4 ("NormalFloat 4")* is a 4-bit
  format whose 16 levels are placed to suit **normally distributed** values, and neural-network weights are roughly normal.
  That puts more levels where most weights are, so the rounding error is lower than with evenly spaced 4-bit levels.
- **Double quantisation**: the per-block scale factors are quantised too, which saves roughly another 0.4 bits per weight.
- **Compute in fp16**: weights are de-quantised on the fly to fp16 for each matrix multiplication. The T4 (Turing
  architecture) has no native bf16 support, which is why fp16 is used.
- **Only the LoRA matrices are trained** (rank 16), so gradients and Adam state exist only for this small set.

**Why rank 16, alpha 32, all-linear?** These are well-established QLoRA defaults (alpha = 2 x r is a common
convention, which gives a scale factor of 2). "All linear layers" (attention q/k/v/o + MLP gate/up/down) works better than
attention-only in the QLoRA paper's results. They were **not tuned on eval**: they were fixed in advance to avoid
overfitting the test set through hyperparameter search.

#### 7.1.3 Gradient checkpointing

During the backward pass you normally keep every layer's activations in memory. **Gradient checkpointing**
throws most of them away and **recomputes** them when needed. That costs about 30% more compute and saves a lot of memory.
It is a standard trade of compute for memory, and it is what lets batch 4 with sequences up to 1,024 tokens fit.

#### 7.1.4 Gradient accumulation

Batch 4 fits in memory and batch 16 does not. **Gradient accumulation** adds up gradients over 4 small
batches before taking one optimiser step, which is mathematically close to a batch of 16. That gives stabler updates
without more memory.

#### 7.1.5 Loss masking (train only on the answer)

The prompt is about 195 tokens and the same for every example. The answer is short. If you computed loss on the prompt,
most of the training signal would go into learning to repeat the instructions. Setting prompt labels to `-100`
puts **100% of the learning signal on producing the answer**. Learning is faster and better aimed.

#### 7.1.6 Early stopping by dev loss

The checkpoint is chosen by **dev loss only**, measured after each epoch. Epoch 1 had the lowest dev loss, so the
later epochs (which started to overfit) were discarded. The eval set was never used to choose anything.

#### 7.1.7 Other training choices

- **Cosine LR schedule + 5% warmup**: a gentle start avoids early instability, and a smooth decay helps the model settle.
- **Single T4 instead of two**: at this size, splitting across two GPUs adds communication overhead and no speed.
- **Dynamic padding** in the custom collator: each batch is padded only to its own longest sequence, not to 1,024,
  so compute isn't wasted on padding.
- **Smoke test**: 20 minutes to catch a bug before spending a 30-minute run (and limited free GPU quota).

### 7.2 Model compression: quantisation

#### 7.2.1 Why quantise at all?

On a CPU, running a model has two phases:

- **Prompt processing ("prefill")**: all prompt tokens are processed together as a big matrix-matrix
  multiplication. This is **compute-bound**: limited by how fast the CPU does arithmetic.
- **Generation ("decode")**: one token at a time. **For every single output token, every weight must be read from RAM**
  once. The arithmetic per weight is tiny, so the CPU spends most of its time waiting for memory. This is
  **memory-bandwidth-bound**.

So the time to write one token ≈ (model size in bytes) / (RAM bandwidth). **Halve the file and you roughly double
the speed of writing.** The project confirmed this: *writing speed x file size* comes out at about 29–36 GB/s for every level,
which is about the laptop's real memory bandwidth. For example, Q4_K_M: 16.1 tok/s x 1.92 GB ≈ 31 GB/s.

Quantisation also directly cuts **RAM**: peak RAM was measured as **file size + about 0.6 GB** at every level (the 0.6 GB is the
KV cache, compute buffers and the runtime).

#### 7.2.2 The levels and how they store weights

| Level | What it is | Real bits/weight here | File |
|---|---|---|---|
| **F16** | Plain 16-bit float. No quantisation. The reference | 16.02 | 6.16 GB |
| **Q8_0** | Blocks of 32 weights; each block has one 16-bit scale and 32 8-bit integers. `w ≈ scale x q` | 8.52 | 3.28 GB |
| **Q5_K_M** | K-quant, mostly 5-bit, "M" (medium) mix | 5.76 | 2.21 GB |
| **Q4_K_M** | K-quant, mostly 4-bit, M mix | 4.98 | 1.92 GB |
| **Q3_K_M** | K-quant, mostly 3-bit, M mix | 4.09 | 1.57 GB |

Why Q8_0 is 8.5 bits and not 8: each block of 32 weights stores 32 bytes plus a 2-byte scale, which is 34 bytes per 32 weights
= 8.5 bits per weight. Every quantisation format pays this "metadata" cost.

The remaining 0.02 bits in F16 comes from 73 small tensors (layer norms) that are always kept in F32.

#### 7.2.3 K-quants and the "_M" mix: why "Q4" is really about 5 bits

**K-quants** (llama.cpp's `Q2_K`…`Q6_K`) use a two-level block structure:

- A **super-block of 256 weights** is split into 8 **sub-blocks of 32**.
- Each sub-block has its own scale (and minimum), but those sub-block scales are themselves **quantised** (to 6 bits)
  relative to one super-block scale in fp16.
- The result is fine-grained scaling (good accuracy) with little metadata overhead. A plain Q4_K tensor is about 4.5 bits per weight.

The **"_M" (medium) recipe** doesn't quantise every tensor the same way. It keeps the **more sensitive tensors at
higher precision**. The type mix recorded in `week4_quant_info.json` shows this:

| Level | Tensors by format |
|---|---|
| Q5_K_M | 216 x Q5_K, **37 x Q6_K**, 73 x F32 |
| Q4_K_M | 216 x Q4_K, **37 x Q6_K**, 73 x F32 |
| Q3_K_M | 144 x Q3_K, 104 x Q4_K, 4 x Q5_K, 1 x Q6_K, 73 x F32 |

Among the Q6_K tensors is the **token embedding**, which SmolLM3 shares between input and output ("tied
embeddings"). It is a large matrix (vocabulary x hidden size) and is sensitive, because it decides which token is written.
Keeping it and the other upgraded tensors at 6 bits is why "Q4_K_M" averages **4.98** bits rather than 4.5.
**Interview point: a quantisation's name is not its real size. Measure bits per weight as file bytes x 8 / number
of weights.**

#### 7.2.4 Why quantise from F16 and not from the 4-bit training model?

- The QLoRA base was in NF4, a bitsandbytes training format that llama.cpp doesn't use. Exporting from it would stack
  two rounding steps (NF4 and then the target format).
- So the adapter was merged into a **full-precision** base, saved as bf16, converted to **F16**, and *every*
  K-level was made from that **same F16** file. All levels share one parent, so any differences come from
  the quantisation alone.

#### 7.2.5 Importance matrix (imatrix): not used, on purpose

llama.cpp can take an **importance matrix**: you run calibration text through the model, record which weights matter most,
and spend the rounding budget where it hurts least. It usually helps low-bit levels (Q3/Q4) noticeably. This project
used **plain quantisation** to keep the comparison simple and the levels comparable. It is also the most obvious
**next step** (section 11), and Q4_K_M's weakness on `escalate_oncall` might well shrink with an imatrix built from log data.

#### 7.2.6 What quantisation did to accuracy

- **F16 = Q8_0 = Q5_K_M within noise.** Q5_K_M is 64% smaller with no measurable loss.
- **Q4_K_M** is where the first damage shows, and it lands on the rarest, most critical class:
  `escalate_oncall` recall 3/6 (vs 6/6), macro-F1 80.5% (vs 88–89%).
- **Q3_K_M** drops on every field (action 93.9%). All 10 extra errors (compared with F16) are on BGL logs.
- **False alarms rise** as bits fall: `escalate_oncall` precision 0.50 → 0.38 → 0.27.
- **Why the rare class breaks first**: a rare class depends on a few subtle signals the model learned from only 11
  examples. Its "margin" (how much more likely the right answer is than the wrong one) is small, so a little rounding
  noise flips it. Common classes have large margins and survive.

### 7.3 Inference and serving optimisation

#### 7.3.1 Prompt prefix caching

Every request starts with the **same ~190-token system prompt**; only the log line changes. Ollama
(llama.cpp) **reuses the computed state (KV cache) of the shared prefix** from the previous request, so only the
new tokens need processing. `bench.py` records `prompt_eval_cached_count` and computes prompt speed on the
*new* tokens only. This is a big win when a long fixed instruction is followed by a short variable input, and
it is the default both in scoring and in serving, so the numbers are consistent.

#### 7.3.2 Choosing the level: a speed surprise

Prompt reading did **not** follow file size. Q4_K_M read prompts at **57 tok/s** against 29–37 for the
rest, which gives it the **fastest first token (0.63 s vs 1.11 s for Q5_K_M)**. The reason: on this CPU, llama.cpp
**repacks Q4_K weights into a layout optimised for the CPU's SIMD instructions**, which speeds up the compute-bound
prefill. It doesn't do this for Q3_K, Q5_K, Q6_K or Q8_0. As a result:

- Q3_K_M *writes* faster (smaller file), but Q4_K_M *reads* faster. Their total time per answer is the **same**
  (3.77 s vs 3.81 s). **Q3_K_M saves only 0.34 GB, loses accuracy and gains no speed, so it is never worth it.**
- **Lesson: you can't predict speed from bit-width alone. Measure on the target hardware.**

#### 7.3.3 Concurrency and throughput (`loadtest.py`)

Ollama was started with `OLLAMA_NUM_PARALLEL=4`, which lets llama.cpp **batch up to 4 requests together**.
Batching helps because decode is memory-bound: one pass over the weights can serve several requests at once.

| Level | Concurrent | Lines/min | Median latency | First token |
|---|---|---|---|---|
| Q4_K_M | 1 → 2 → 4 | 15.0 → 20.6 → 27.4 (+82%) | 3.6 → 5.4 → 8.0 s | 0.64 → 0.97 s |
| Q5_K_M | 1 → 2 → 4 | 12.5 → 16.6 → 18.3 (+46%) | 4.2 → 6.6 → 11.8 s | 1.17 → 1.45 s |

- **This is the classic throughput-vs-latency trade-off**: more concurrency gets more total work done, but each
  request waits longer.
- Q4_K_M scales better (likely the same prefill speed-up; not proven).
- **Q5_K_M: 2 concurrent is the sensible limit on this laptop.** Going to 4 adds only 10% throughput, while the
  median wait rises from 6.6 s to 11.8 s.
- Allowing 4 parallel slots costs a single request no measurable speed, only about 0.4 GB of extra RAM (KV cache
  for each slot).
- **Determinism caveat**: under batching, a few answers changed (1/5/7 out of 164 for Q4_K_M at c=1/2/4), because batched
  floating-point maths rounds differently. llama.cpp documents this. Accuracy did not measurably change.

#### 7.3.4 Small serving optimisations that matter

- **`keep_alive: 15m`**: the model stays in RAM. A cold load costs seconds on every request otherwise.
- **`127.0.0.1` instead of `localhost`**: on Windows, `localhost` resolves to IPv6 `::1` first. Ollama listens only on
  IPv4, and Windows retries the refused connection for **about 2 s per request**. This was found during benchmarking and fixed with one
  line. It is a good debugging story for an interview.
- **Async FastAPI + httpx**: the API doesn't block a thread while waiting for the model.
- **Input size limit** (2,000 chars) protects the context window and keeps latency bounded.
- **Slim Docker image**: `python:3.13-slim`, only the needed files, `requirements-serve.txt` with just 3 packages.

### 7.4 Measuring optimisation properly (the benchmark methodology)

An optimisation claim is only as good as its measurement. `bench.py` is careful in these ways:

- **Same runtime for both axes**: accuracy and cost both come from Ollama 0.34.2 on the same laptop. (The original
  plan was to benchmark with llama.cpp directly. That was switched to Ollama so the two numbers can't disagree
  just because two different programs were used.)
- **Same prompt and settings as scoring**, and every benchmark answer was checked **character for character against the scored
  answer** (all 5 x 328 matched). The measured cost therefore belongs to exactly the outputs that were scored.
- **Isolation**: all models unloaded before each run; only one `llama-server` process allowed; Docker must be
  closed.
- **Machine-state checks**: Ollama version, charger connected, Windows power mode = "Best performance", enough free RAM,
  sleep prevented. Laptops throttle when on battery or in power-saving mode, so these checks matter.
- **Cold vs warm**: 3 fresh loads are timed separately; 3 warm-up requests are discarded; then 328 measured requests.
- **Streaming** so that **time to first token (TTFT)** can be measured as well as total latency.
- **Peak RAM** of the `llama-server` process itself (via psutil), not the whole machine.
- **Percentiles**: median (p50) and p95, not just averages. Tail latency is what users notice.
- **Noise floor measured**: 30 lines were re-run at the end of the session. Whole-answer time moved 0.7% and TTFT 6.8%, so smaller gaps
  between levels are treated as noise.
- **Units stated**: GB = 10^9 bytes (so the figures look about 7% bigger than Windows Task Manager's 2^30-byte GB).

### 7.5 Cost results and the recommendation

| Level | Peak RAM | First token | Median answer | Lines/min | Action acc | Macro-F1 |
|---|---|---|---|---|---|---|
| Q3_K_M | 2.17 GB | 0.96 s | 3.77 s | 14.2 | 93.9% | 77.7% |
| Q4_K_M | 2.51 GB | **0.63 s** | 3.81 s | 14.1 | 96.0% | 80.5% |
| **Q5_K_M** | 2.80 GB | 1.11 s | 4.50 s | 11.9 | **97.3%** | **89.5%** |
| Q8_0 | 3.87 GB | 1.21 s | 5.97 s | 9.0 | 97.0% | 88.2% |
| F16 | 6.75 GB | 1.10 s | 9.91 s | 5.5 | 97.0% | 88.2% |

**Recommendation**: **Q5_K_M by default.** It is as accurate as F16 at 41% of the RAM and 2.2x the speed. Use **Q4_K_M** when a
fast first token or higher throughput matters more, accepting the warning sign on `escalate_oncall`. **Never
use Q3_K_M or Q8_0/F16**: the first is dominated by Q4_K_M, and the others cost more and buy no accuracy.

---

## 8. Evaluation methodology

### 8.1 Three baselines

1. **Most-common answer** (always `no_action`): 85.7% action accuracy, because 86% of eval is `no_action`. **Any model must
   beat this.**
2. **Untuned SmolLM3-3B** (same prompt, Q8_0): 83.3% action accuracy. **That is worse than always saying no_action**, and it found 0/16
   check_network and 0/6 escalate_oncall. Its strict JSON rate was 0%.
3. **Fine-tuned at F16**: the upper reference for the quantisation comparison.

### 8.2 Parsing rules (`score.parse`)

- **Strict JSON**: the whole output is exactly one JSON object.
- **Lenient** (used for field scores): remove any `<think>…</think>` and markdown ``` fences, then parse from the
  first `{` to the last `}`. This is fair to models that wrap JSON in extra text.
- **Unparseable or missing output = wrong on every field.** It is never skipped. Skipping failures would inflate scores.
- **Schema-valid**: all 4 keys, strings where strings are expected, identifiers a list of strings.

### 8.3 Field metrics

- Categorical fields (service, severity, action): exact match after trimming and ignoring case.
- Identifiers: **multiset** comparison (order ignored, duplicates counted), with micro-averaged precision, recall and F1, plus an
  "exactly right" rate.
- Everything is broken down by **seen vs unseen pattern** and **by system** (HDFS/BGL/OpenStack/Spark).

### 8.4 Why macro-F1 is the headline metric

The data is very imbalanced (282 of 329 are `no_action`). **Accuracy rewards ignoring the rare classes.** Macro-F1 gives
each of the 4 classes equal weight, so missing `escalate_oncall` hurts a lot. That is the right incentive for a triage tool,
where the rare lines are the ones that matter. Untuned 24.3% → fine-tuned 88.2% is the real improvement,
and it shows much more clearly than 83.3% → 97.0% accuracy.

### 8.5 Confidence intervals (bootstrap)

With only 329 examples (and 6 escalate_oncall), a 1-point difference may be luck. `score.py` uses the **bootstrap**:
resample the 329 eval rows *with replacement* 1,000 times (fixed seed), recompute every metric each time, and take
the 2.5th and 97.5th percentiles as a **95% confidence interval**. Example: Q5_K_M action accuracy 97.3% has a CI of
[95.4%, 98.8%], and macro-F1 89.5% has a CI of [80.7%, 96.0%]. That macro-F1 interval is wide, because it rests on 22 rare lines.

That is why the README says the Q4/Q3 drop is **"suggestive, not proven"**: the intervals overlap. Also, these are
**per-model** intervals, **not a paired test** of the difference (a paired bootstrap or McNemar's test on the same
lines would be the stricter next step).

### 8.6 Avoiding test-set contamination (a checklist to recite)

- The split is by pattern, with unseen patterns held out entirely.
- Distinct-text deduplication across splits, enforced by asserts.
- The eval set is frozen and hashed before training, and the hash is checked before every scoring run.
- Label vocabulary and hyperparameters were fixed before freezing and training.
- The checkpoint is chosen by dev loss only.
- Every model is scored with the same prompt, settings, runtime and machine.

---

## 9. Results and what they mean

### 9.1 Does fine-tuning work? Yes, decisively (Q8_0, frozen eval set, n=329)

| Metric | Untuned | Fine-tuned | Always no_action |
|---|---|---|---|
| Strict JSON | 0.0% | 99.7% | – |
| Action accuracy | 83.3% | 97.0% | 85.7% |
| Action macro-F1 | 24.3% | 88.2% | 23.1% |
| Severity | 66.6% | 95.7% | 84.5% |
| Service | 20.1% | 79.9% | 23.1% |
| Identifier F1 | 29.0% | 92.2% | – |
| check_network found | 0/16 | 16/16 | 0/16 |
| escalate_oncall found | 0/6 | 6/6 | 0/6 |

### 9.2 Seen vs unseen patterns (the generalisation test)

| | Seen (185) | Unseen (144) |
|---|---|---|
| Action | 98.4% | 95.1% |
| Severity | 97.8% | 93.1% |
| Service | 95.7% | **59.7%** |
| Identifier F1 | 96.8% | 87.9% |

Action, severity and identifiers generalise well. **Service collapses on unseen patterns** because the component name
**is not in the input text**: for a new pattern there is nothing to copy, and the model can only guess from the wording.
That is a limit of the task, not a training failure.

### 9.3 A robustness finding

Ollama updated itself partway through (0.32.6 → 0.34.2). Both models were re-scored. The same inputs changed **262 of 329
untuned outputs but only 13 of 329 fine-tuned ones**, and no conclusion changed. Fine-tuning made the model far more
**stable** as well as more accurate. Lesson: **pin your runtime version** (the Docker image is pinned to 0.34.2).

### 9.4 The headline in one sentence

"Fine-tuning a 3B model took action macro-F1 from 24% to 88%. Compressing it to about 5.8 bits per weight (Q5_K_M) kept all
of that, at 64% less size and 2.2x the speed on a laptop CPU. The first damage appears at about 5 bits (Q4_K_M), and it
hits the rarest, most critical class first."

---

## 10. Engineering practices worth mentioning

- **Single source of truth** (`config.py`). The prompt, vocabulary, seed and paths are defined once and imported everywhere, so they can't drift
  between training, scoring, benchmarking and serving.
- **Reproducibility**: fixed seed 42 everywhere; pinned library versions in the notebooks and requirements; a pinned
  llama.cpp commit; a pinned Ollama version; SHA-256 hashes of the data and every model file; run metadata saved
  (`run_info.json`, `week4_quant_info.json`).
- **Fail loudly, never silently**: scripts `die()` with a clear message and write nothing when a check fails (leaks,
  hash mismatch, wrong Ollama version, missing model, NaN loss, metadata drift).
- **Never overwrite results**: new runs use `--tag`/`--suffix`, and benchmark scripts refuse to run if the output exists.
- **Resumable long jobs**: teacher labelling and inference skip finished items.
- **Smoke tests before expensive runs**: `SMOKE_TEST` in the notebook, `--limit` and `--suffix smoke` in the scripts.
- **Human in the loop** where judgement is needed (the eval label review), with the corrections applied everywhere.
- **Security basics**: API keys come from environment variables or Kaggle Secrets and never from code; `.env` is git-ignored; the container
  runs as non-root; the model directory is mounted read-only.
- **Honest reporting**: confidence intervals, noise floor, "suggestive not proven", limits stated up front.

---

## 11. Limitations

Say these yourself. It shows maturity.

1. **Service is memorisation.** The component isn't in the input, so 59.7% on unseen patterns is about the ceiling of the
   setup. More training won't fix it; you'd need to give the model the header or a lookup.
2. **The rare classes are tiny.** `escalate_oncall` has 6 eval lines and 11 distinct training lines. Its results are anecdotal.
   Precision is only 0.50 even at the best levels (it raises false alarms).
3. **check_network was learned from HDFS only.** Every training example comes from one system.
4. **LogHub-2k is small and repetitive.** 120 of 213 patterns have only one distinct message, which limits eval size.
5. **The recommended_action labels come from a teacher LLM** (human-reviewed for eval), so they are not real on-call decisions.
6. **One eval line is unscorable**: its correct answer is longer than the 384-token output limit, so it counts as wrong for every
   model.
7. **Per-model CIs, not paired tests**, so the differences between levels are not statistically proven.
8. **One machine** (an Intel laptop CPU, Windows). Speed rankings, especially the Q4_K prefill trick, may differ on other hardware.
9. **No importance matrix**: the low-bit levels may be pessimistic.

**Natural next steps** (suggestions, not done in the project): imatrix quantisation calibrated on log text; a paired
bootstrap or McNemar test between levels; a bigger and more diverse eval set (more LogHub systems, real incidents);
constrained or grammar decoding for guaranteed JSON; testing on ARM and GPU; giving the model the log header for `service`.

---

## 12. Interview questions and model answers

**Q: Explain your project in 30 seconds.**
A: I fine-tuned a 3-billion-parameter model, SmolLM3, to turn a raw server log line into a structured JSON
triage record: component, severity, IDs, and a recommended action. Then I compressed it to five precision levels and
measured accuracy against cost on a laptop CPU. Fine-tuning took macro-F1 from 24% to 88%. The 5-bit level kept all of it
at 64% less memory and 2.2x the speed, and I served it with FastAPI, Ollama and Docker.

**Q: Why a small model instead of calling GPT-4?**
A: Cost per call, latency, privacy (logs often contain internal IPs and IDs), and it works offline. The
question was whether a small, cheap model can be good enough, and for this narrow task it is.

**Q: Why fine-tune instead of prompt engineering?**
A: The untuned model with the same prompt scored *below* the always-no_action baseline (83.3% vs 85.7%), produced 0% strict
JSON, and found none of the rare classes. Fine-tuning taught it the format and the labels. Strict JSON went to 99.7%.

**Q: What is LoRA and why did you use QLoRA?**
A: LoRA freezes the model and trains small low-rank matrices next to each layer, a tiny fraction of the
parameters. QLoRA also stores the frozen base in 4-bit NF4, so a 3B model trains on a free 16 GB T4. Rank 16, alpha 32,
all linear layers, LR 2e-4: standard defaults fixed in advance.

**Q: How did you prevent data leakage?**
A: I split by log pattern, not by line, and held out 15% of patterns entirely. I deduplicated texts across splits with asserts,
froze and hashed the eval set before training and verified the hash before every scoring run, and chose the checkpoint by dev loss only.

**Q: Your accuracy is 97%. Isn't that just the imbalance?**
A: Exactly the right worry. Always answering no_action gives 85.7%. That's why I report macro-F1, which weights all four classes
equally: 23.1% for that baseline, 88.2% for my model. I also report per-class recall: 16/16 check_network, 6/6 escalate_oncall.

**Q: What is quantisation and why does it make inference faster?**
A: It stores weights in fewer bits. On a CPU, generating each token means reading every weight from RAM, so it is
memory-bandwidth-bound. A smaller file means fewer bytes per token and faster generation. I confirmed it: tokens per second x file
size is about constant, roughly 30 GB/s, which is the laptop's memory bandwidth. It also cuts RAM to about the file size plus 0.6 GB.

**Q: What's the difference between Q8_0 and Q4_K_M?**
A: Q8_0 uses blocks of 32 weights, one fp16 scale each, and 8-bit integers, which works out to 8.5 bits. K-quants use 256-weight super-blocks with
quantised sub-block scales, and the "_M" recipe keeps sensitive tensors such as the tied embedding at Q6_K. So Q4_K_M is really about 5 bits per weight,
not 4. I measured it from file bytes.

**Q: Which level would you deploy and why?**
A: Q5_K_M. It's statistically the same as F16 on every field, at 41% of the RAM and 2.2x the speed. Q4_K_M if first-token latency
or throughput matters most, but it found only 3 of 6 escalate_oncall lines, and that's the class you least want to miss.

**Q: Something surprising you found?**
A: Q4_K_M had the fastest first token, faster even than the smaller Q3_K_M, because llama.cpp repacks Q4_K weights into a
SIMD-friendly layout on this CPU, which speeds up prompt processing. So Q3_K_M saves 0.34 GB but is no faster and is
less accurate. You can't predict speed from bit-width; you have to measure on the target hardware.

**Q: How did you make sure the benchmark measured the same model you scored?**
A: Same Ollama version, same machine, same prompt function, settings and parser, imported rather than copied. Every one
of the 1,640 benchmark answers was character-for-character identical to the scored answer.

**Q: How did you handle concurrency?**
A: I set `OLLAMA_NUM_PARALLEL=4` and ran closed-loop load tests at 1, 2 and 4 concurrent users. Throughput rose 82% for Q4_K_M at 4
users, but median latency more than doubled. That's the throughput-latency trade-off. For Q5_K_M, 2 concurrent requests is the sweet spot on a laptop.

**Q: Why are outputs not identical under load?**
A: Batching changes the order of floating-point operations, and float addition isn't associative, so rounding differs slightly. A few
long identifier lists changed. Accuracy didn't measurably change.

**Q: Why Ollama instead of llama.cpp directly or vLLM?**
A: Ollama wraps llama.cpp (the best CPU runtime), so I get the same kernels with a simple API and model management.
vLLM targets GPUs. Most importantly, I used one runtime for both accuracy and cost, so the two axes can't disagree because of
different software.

**Q: Why raw mode and temperature 0?**
A: Raw mode sends my hand-written chat template exactly, so Ollama can't apply a different template. Temperature 0 with a fixed
seed gives deterministic greedy decoding, so differences between models come from the models and not from sampling luck. I didn't
force JSON either, because producing valid JSON is part of what's measured.

**Q: How do you know the differences between levels are real?**
A: I bootstrapped 95% confidence intervals with 1,000 resamples. Q5/Q8/F16 overlap completely. The Q4/Q3 drops are consistent but
their intervals overlap, so I call them suggestive. A paired test would be the next step.

**Q: Biggest limitation?**
A: `service` on unseen patterns (59.7%), because the component name isn't in the input; it's memorisation. And the rare
classes are tiny (6 escalate_oncall lines), so those results are anecdotal.

**Q: What would you do next?**
A: Importance-matrix quantisation calibrated on logs (it would likely fix Q4_K_M's rare-class drop), a paired significance test, a
larger eval set, grammar-constrained decoding, and testing on ARM and GPU.

**Q: Explain your API design.**
A: FastAPI, async with httpx. `POST /triage` validates input (1–2,000 chars), builds the frozen prompt, calls Ollama,
parses with the scoring parser, and returns the record plus a schema-valid flag, raw output, token counts and timings.
`GET /health` checks Ollama and the model. Errors map to clear status codes (503 unreachable, 504 timeout, 502 upstream error).
It is containerised with compose, runs as a non-root user, and the image is slim.

**Q: A debugging story?**
A: Every request had about 2 extra seconds of latency on Windows. The cause: `localhost` resolves to IPv6 first, Ollama only
listens on IPv4, and Windows retries the refused connection. Switching to `127.0.0.1` removed it. I found it because the benchmark
records the "overhead" between my measured time and Ollama's own timer.

**Q: How long did training take, and how did you pick the epoch?**
A: About 30 minutes on one T4, 3 epochs. Dev loss was lowest after epoch 1 (0.0496), so I kept that checkpoint; later epochs
overfit slightly.

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **Adapter** | The small set of trained LoRA weights, applied on top of a frozen base model |
| **Bootstrap** | Resampling the test set with replacement many times to estimate a confidence interval |
| **BOS token** | "Beginning of sequence" marker some tokenizers put at the start of the input |
| **bf16 / fp16** | Two 16-bit float formats. bf16 has a wide range and low precision; fp16 has a narrow range and higher precision |
| **Checkpoint** | A saved snapshot of the model during training |
| **Context window** | The maximum number of tokens (prompt + answer) the model handles at once (2048 here) |
| **Decode / generation** | Producing output tokens one by one. Memory-bandwidth-bound on CPU |
| **Distillation** | Training a small model on labels produced by a bigger one |
| **Epoch** | One full pass over the training data |
| **GGUF** | llama.cpp's single-file model format |
| **Gradient accumulation** | Adding gradients over several mini-batches before one update, to simulate a bigger batch |
| **Gradient checkpointing** | Recomputing activations in the backward pass to save memory |
| **Greedy decoding** | Always picking the most likely next token (temperature 0) |
| **imatrix** | Importance matrix: calibration data that guides quantisation to protect important weights |
| **K-quant** | llama.cpp's super-block quantisation family (Q2_K…Q6_K) |
| **KV cache** | Stored attention keys and values of past tokens, so they aren't recomputed |
| **Loss masking** | Excluding prompt tokens from the loss (label -100) so only the answer is learned |
| **LoRA / QLoRA** | Low-rank adapters / the same with a 4-bit frozen base |
| **Macro-F1** | The average of per-class F1 scores, with every class weighted equally |
| **NF4** | 4-bit NormalFloat, a quantisation format suited to normally distributed weights |
| **Prefill / prompt processing** | Processing all prompt tokens at once. Compute-bound |
| **Prefix caching** | Reusing the KV cache of a prompt prefix shared with the previous request |
| **Quantisation** | Storing weights with fewer bits |
| **Raw mode** | Ollama option that sends the prompt exactly as given, with no template |
| **SFT** | Supervised fine-tuning on (prompt, answer) pairs |
| **Template / pattern (log)** | A log message with its variable parts replaced by `<*>` |
| **Tied embeddings** | The same matrix used for input token embeddings and the output layer |
| **TTFT** | Time to first token: how long until the answer starts appearing |
| **Throughput** | Work completed per unit time (lines per minute here) |
| **p50 / p95** | Median / 95th-percentile latency |
