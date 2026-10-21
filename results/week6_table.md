# Week 6 - single-stream benchmark results

| Level | File (GB) | RAM after load (GB) | Peak RAM (GB) | First token p50 / p95 (s) | Whole answer p50 / p95 (s) | Lines per min | Writing (tok/s) | Prompt reading (tok/s) | Reload (s) | Answers same as Week 4 |
|---|---|---|---|---|---|---|---|---|---|---|
| Q3_K_M | 1.57 | 1.83 | 2.17 | 0.96 / 1.94 | 3.77 / 6.88 | 14.21 | 18.19 | 37.13 | 1.72 | 328/328 |
| Q4_K_M | 1.92 | 2.17 | 2.51 | 0.63 / 1.23 | 3.81 / 6.84 | 14.07 | 16.11 | 57.0 | 2.23 | 328/328 |
| Q5_K_M | 2.21 | 2.47 | 2.8 | 1.11 / 2.24 | 4.5 / 8.22 | 11.88 | 14.91 | 32.19 | 1.72 | 328/328 |
| Q8_0 | 3.28 | 3.53 | 3.87 | 1.21 / 2.55 | 5.97 / 10.79 | 8.98 | 10.65 | 29.16 | 2.23 | 328/328 |
| F16 | 6.16 | 6.41 | 6.75 | 1.1 / 2.2 | 9.91 / 17.36 | 5.54 | 5.83 | 32.5 | 6.27 | 328/328 |

## Conditions
- Runs: 2026-10-12T00:36 to 2026-10-12T03:58, one level at a time, 5-minute pause between levels.
- Ollama 0.34.2 (checked before and after every run), model run by llama-server with 8 threads (chosen automatically, not tuned), CPU only (0 MB on GPU for every level).
- Laptop: 12 cores / 16 threads, 16.79 GB RAM, power mode 'Best performance', on charger: True. Docker and other apps closed.
- Requests: the exact scoring prompt and settings ({"temperature": 0, "seed": 42, "num_ctx": 2048, "num_predict": 384, "stop": ["<|im_end|>"]}, raw mode), one at a time, streamed.
- Warm numbers: 328 eval lines per level (the 1 line over 2000 characters, r01580, is skipped, as in the API), after 3 warm-up dev lines that are not counted.
- Ollama re-uses the shared start of the prompt between requests: on average 178.64 of 219.06 prompt tokens were re-used, so only the log-specific part is read each time. This is the default and was also on during scoring.
- Reload = median of 2 fresh loads with the model file already in Windows' memory (not from disk).
- Percentiles: linear interpolation (numpy's default). Time outside Ollama (connecting, reading the answer): about 16.95 ms per request.

## Drift check (the same first 30 Q3_K_M lines, start vs end of the session)
- Writing speed +0.4%, prompt reading +3.0%, first token -6.8%, whole answer -0.7%.
- Differences between levels smaller than these changes are within run-to-run noise.

## Notes
- The benchmark connects to Ollama at 127.0.0.1, not localhost: on this Windows PC 'localhost' added about 2.05 s to every request (Windows retries the refused IPv6 connection). Found in the smoke test and fixed before the real runs.
- Q8_0: the file Ollama runs (sha256 4a33056afbc46dc7424b8de368c004574265f2303f7b02ea9ef5a143e5f970b6) has a different hash from the recorded Week 3 file (8755be93df8f769947eb4eef6596909c7437f5f6b60cfcc5c4ce5988ca37155d) only because its tensors are stored in a different order. Checked 2026-10-15: all 27 settings and all 326 tensors are byte-identical. Every other level ran exactly the Week 4 file.
- Memory: llama-server copies the whole model into RAM (Ollama turns off memory-mapping when there is no GPU). RAM grew during each run from 'after load' to 'peak' in identical 43 MB steps at the same lines for every level, so the growth depends on the prompts, not on the weights. Its cause was not confirmed.
- Writing speed x file size is about the same for every level (memory bandwidth limit). Q4_K_M reads prompts fastest because llama.cpp rearranges Q4_K weights into a faster layout on this CPU and does not do that for Q3_K, Q5_K, Q6_K or Q8_0.
