# wip/backend-sampler-reserve-once — backend sampling: reserve the compute buffers once per sequence, not on every `set_sampler()`

One `git am` file (one commit), made on `v15-6050b467c-r167` (applied tree `95022c9a`).  It touches
`src/llama-context.cpp` and `src/llama-context.h` (+33 / -4).
`backend-sampler-reserve-once.patch` sha256 `79ea4f508f30ec4977dfceae2de150cdd40d52ffcc71c42ca50d9c7ddaf64ac7`.

**The defect is llama.cpp's own.**  `src/llama-context.cpp` on `ggml-org/llama.cpp` master (checked 2026-10-29)
behaves the same, and `6050b467c` (this delivery's base) too, so a copy cut against plain `6050b467c`, without the
kill switch, is in `upstream/backend-sampler-reserve-once/` (README + `UPSTREAM-PR-backend-sampler-reserve-once.patch`),
following the `upstream/` convention.  Whether and when it goes to ggml-org is yours to decide.

**Why.** `llama_context::set_sampler()` sets `sched_need_reserve = true` every time an offloadable sampler chain is
set for a sequence, and every time one is removed.  The server sets a new chain for every request with
`--backend-sampling`, and removes it for a request whose chain cannot be offloaded (a grammar).  `sched_reserve()`
re-creates the scheduler, so each such request frees and allocates all compute buffers again - at the same size -
before its first token.

**What changes.**  `sched_reserve()` records the sequences that had a backend sampler when the buffers were reserved
(`sampling.reserved`).  `set_sampler()` asks for a reserve only when a sequence gets a backend sampler that the
current buffers were not reserved with; setting a chain again, or removing one, asks for none.  That is safe because
the graph is rebuilt anyway (the samplers are part of the graph-reuse check), the output buffers are sized per
decode (`output_reserve()`), and the graph allocator grows a buffer on demand if a chain ever needs more than the
reserved one.  The `-sm tensor` branch (no backend sampling there) is unchanged.  No arithmetic changes.

**Kill switch.**  `LLAMA_BACKEND_SAMPLER_RESERVE_ONCE=0` restores the reserve on every set / removal (fork version
only; the upstream copy has no switch).

## Results

Linux: Ryzen 9 3900X, one R9700 (gfx1201), Ubuntu 24.04.5, ROCm 10.0.0 (TheRock pip).  Qwen3.8-27B-UD-Q4_K_XL alone
(no draft), `--backend-sampling -ngl 99 -c 65536 -ub 2048 -b 2048 --parallel 2 --kv-unified -fa on -ctk q8_0
-ctv q8_0`, `ROCBLAS_USE_HIPBLASLT=0` (GREEDY-PURITY §44, for the text comparison).  Per server start: a warm-up,
six sampled `/completion` requests (1.5-2K-token prompts, `cache_prompt: false`, temperature 0.7, top-k 40, top-p
0.95, a seed each - an offloadable chain), one request with a grammar (its chain is not offloadable, so the
sampler is removed), one more sampled request.  Two starts per arm, interleaved (R1 R0 R0 R1, then P U U P).

| | before | after | |
|---|---|---|---|
| this fork, r167 + the patch: `=0` vs default | 1560 ms (prompt 989 ms) | 1137 ms (prompt 576 ms) | **-423 ms** per sampled request |
| plain llama.cpp `6050b467c`: without vs with the upstream copy | 1420 ms (prompt 1090 ms) | 978 ms (prompt 636 ms) | **-442 ms** per sampled request |
| the grammar request (fork / plain) | 814 / 795 ms | 443 / 396 ms | about -0.4 s |

Median client time to first token over the sampled requests (prompt ms from the server's timings in brackets; the
reserve runs inside the request's first decode, so it lands in prompt ms).  Every request gave the same text in
both arms of each pair.  The server's default log level does not print `sched_reserve`'s "reserving" line, so the
reserves were not counted from the log; the step in time to first token is the evidence.

**Not covered:** Windows; multi-GPU (the `-sm tensor` path is unchanged); a draft context with
`--spec-draft-backend-sampling` (same function, not measured); a chain that needs larger buffers than the one the
buffers were reserved with (the allocator's on-demand growth is relied on, not exercised here).

*Written by Claude, an AI model by Anthropic, from measurements on the contributor's machine.*
