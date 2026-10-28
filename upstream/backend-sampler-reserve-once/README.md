# backend-sampler-reserve-once — llama: backend sampling: reserve once per sequence, not on every `set_sampler()`

`UPSTREAM-PR-backend-sampler-reserve-once.patch` is cut against plain `ggml-org/llama.cpp` **`6050b467c`** (the
delivery base; applied tree `56f86356`), sha256 `c2f3d0b6c34d3f27f167d23998ffd63694ec90680c4c069987ba239bbc4a0cbd`.
It touches `src/llama-context.cpp` and `src/llama-context.h` (+19 / -7).  It is the fork version in
`wip/backend-sampler-reserve-once/` without the `LLAMA_BACKEND_SAMPLER_RESERVE_ONCE` kill switch.

**What.** `llama_context::set_sampler()` sets `sched_need_reserve = true` whenever an offloadable sampler chain is set
for a sequence and whenever one is removed; `llama-server --backend-sampling` sets a new chain for every request (and
removes it for a grammar request), so every request re-creates the scheduler and frees / allocates all compute
buffers again at the same size before its first token.  The patch records the sequences the buffers were reserved
with (`sampling.reserved`, filled in `sched_reserve()`) and asks for a reserve only when a sequence gets a backend
sampler the current buffers were not reserved with.  A chain set again or removed needs none: the graph is rebuilt
anyway (samplers are part of the reuse check), the output buffers are sized per decode, and the graph allocator
grows a buffer on demand.  The tensor-split branch is unchanged.  ggml-org master behaves the same as `6050b467c`
here (checked 2026-10-29).

**Validation** (on `6050b467c`, gfx1201 R9700, Linux, ROCm 10.0.0): plain vs patched `llama-server`, Qwen3.8-27B-UD-
Q4_K_XL, `--backend-sampling -c 65536 -ub 2048 --parallel 2 --kv-unified -fa on -ctk q8_0 -ctv q8_0`,
`ROCBLAS_USE_HIPBLASLT=0`; six sampled requests (temperature 0.7, top-k 40, top-p 0.95, seeded, 1.5-2K-token prompts),
a grammar request and one more sampled request per start, two starts each, P U U P.  Median time to first token
1420 -> 978 ms (-442 ms; prompt ms 1090 -> 636), the grammar request 795 -> 396 ms; identical text for every request.

**Not validated:** other backends (CUDA, Vulkan, Metal) - the change is in `llama_context`, not in a backend; Windows;
multi-GPU; a chain that needs more buffer than the reserved one (relies on the allocator's growth).  Re-run the
validation on current master before filing.

*Written by Claude, an AI model by Anthropic, from measurements on the contributor's machine.*
