# Week 7 - load test (several requests at once)

| Level | At once | Lines per min | Output tok/s (total) | Whole answer p50 / p95 (s) | First token p50 / p95 (s) | Writing tok/s per request | Peak RAM (GB) | Answers same as Week 4 |
|---|---|---|---|---|---|---|---|---|
| Q4_K_M | 1 (Week 6, -np 1) | 14.64 | - | 3.66 / 6.81 | 0.63 / 1.21 | 16.13 | - | - |
| Q4_K_M | 1 | 15.02 | 13.52 | 3.55 / 6.64 | 0.64 / 1.17 | 16.7 | 2.79 | 163/164 |
| Q4_K_M | 2 | 20.58 | 18.54 | 5.42 / 9.5 | 0.78 / 1.33 | 10.93 | 2.95 | 159/164 |
| Q4_K_M | 4 | 27.38 | 24.59 | 7.98 / 14.15 | 0.97 / 2.18 | 7.1 | 3.24 | 157/164 |
| Q5_K_M | 1 (Week 6, -np 1) | 12.41 | - | 4.32 / 8.2 | 1.11 / 2.12 | 14.93 | - | - |
| Q5_K_M | 1 | 12.48 | 11.09 | 4.24 / 8.04 | 1.17 / 2.22 | 15.32 | 2.92 | 164/164 |
| Q5_K_M | 2 | 16.6 | 14.79 | 6.63 / 11.89 | 1.24 / 2.19 | 9.21 | 2.94 | 163/164 |
| Q5_K_M | 4 | 18.25 | 16.27 | 11.82 / 22.43 | 1.45 / 3.95 | 4.7 | 3.36 | 162/164 |

## Did the answers change? (same 164 lines; Week 4 answers vs this run)
| Level | At once | Action acc | Action macro-F1 | Severity acc | Service acc | ID F1 |
|---|---|---|---|---|---|---|
| Q4_K_M | 1 | 98.2 -> 98.2 | 88.0 -> 88.0 | 96.3 -> 96.3 | 79.3 -> 79.3 | 91.6 -> 91.4 |
| Q4_K_M | 2 | 98.2 -> 97.6 | 88.0 -> 87.1 | 96.3 -> 95.7 | 79.3 -> 79.3 | 91.6 -> 90.8 |
| Q4_K_M | 4 | 98.2 -> 98.2 | 88.0 -> 88.0 | 96.3 -> 95.7 | 79.3 -> 79.3 | 91.6 -> 91.4 |
| Q5_K_M | 1 | 99.4 -> 99.4 | 99.0 -> 99.0 | 96.3 -> 96.3 | 79.3 -> 79.3 | 92.6 -> 92.6 |
| Q5_K_M | 2 | 99.4 -> 99.4 | 99.0 -> 99.0 | 96.3 -> 96.3 | 79.3 -> 79.3 | 92.6 -> 92.7 |
| Q5_K_M | 4 | 99.4 -> 99.4 | 99.0 -> 99.0 | 96.3 -> 96.3 | 79.3 -> 79.3 | 92.6 -> 92.3 |

Changed answers (prompt tokens re-used from the previous request: this run vs Week 6):
- Q4_K_M x1 r03224: changed identifiers; action no_action -> no_action (correct: no_action); identifiers 4/10 -> 4/12 correct; re-used 187 vs 174
- Q4_K_M x2 r01865: changed service; action no_action -> no_action (correct: no_action); no field became right or wrong; re-used 174 vs 174
- Q4_K_M x2 r03221: changed identifiers; action no_action -> no_action (correct: no_action); identifiers 4/5 -> 4/11 correct; re-used 174 vs 187
- Q4_K_M x2 r04034: changed severity, recommended_action; action no_action -> investigate (correct: no_action); severity now WRONG; recommended_action now WRONG; re-used 174 vs 174
- Q4_K_M x2 r06016: changed service; action no_action -> no_action (correct: no_action); no field became right or wrong; re-used 174 vs 174
- Q4_K_M x2 r07050: changed identifiers; action no_action -> no_action (correct: no_action); identifiers 2/2 -> 1/2 correct; re-used 174 vs 191
- Q4_K_M x4 r01092: changed service; action no_action -> no_action (correct: no_action); service now WRONG; re-used 174 vs 174
- Q4_K_M x4 r01865: changed service; action no_action -> no_action (correct: no_action); no field became right or wrong; re-used 174 vs 174
- Q4_K_M x4 r02345: changed service; action check_network -> check_network (correct: check_network); service now RIGHT; re-used 174 vs 174
- Q4_K_M x4 r03224: changed identifiers; action no_action -> no_action (correct: no_action); identifiers 4/10 -> 3/9 correct; re-used 174 vs 174
- Q4_K_M x4 r04034: changed severity, recommended_action; action no_action -> investigate (correct: no_action); severity now WRONG; recommended_action now WRONG; re-used 174 vs 174
- Q4_K_M x4 r06003: changed identifiers, recommended_action; action investigate -> no_action (correct: no_action); recommended_action now RIGHT; identifiers 0/2 -> 0/1 correct; re-used 174 vs 174
- Q4_K_M x4 r07040: changed identifiers; action no_action -> no_action (correct: no_action); identifiers 2/2 -> 1/2 correct; re-used 191 vs 191
- Q5_K_M x2 r07009: changed identifiers; action no_action -> no_action (correct: no_action); identifiers 1/1 -> 2/2 correct; re-used 191 vs 191
- Q5_K_M x4 r03221: changed identifiers; action no_action -> no_action (correct: no_action); identifiers 4/5 -> 4/9 correct; re-used 174 vs 187
- Q5_K_M x4 r07009: changed identifiers; action no_action -> no_action (correct: no_action); identifiers 1/1 -> 2/2 correct; re-used 174 vs 191

## Conditions
- 164 eval lines per run (every second eval line of up to 2000 characters), the same lines in every run; exact scoring prompt and settings; streamed; 127.0.0.1.
- Ollama 0.34.2 restarted with OLLAMA_NUM_PARALLEL=4: llama-server -np 4 -c 8192 (4 slots of 2048).
- Closed loop: N 'users' (N = 1, 2, 4); each sends a line, waits for the whole answer, sends the next.
- Each run: fresh load, N warm-up dev lines (not counted), then the measured lines.
- The 'Week 6' rows are the single-stream benchmark (-np 1) on the same lines, for comparison (its lines per minute = 60 / mean time per answer).
- Accuracy in the answer check is on these lines only - not comparable with the 329-line Week 4 figures.
