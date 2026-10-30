# wip/spec-ckpt-device — the target's speculative checkpoint in device memory

*Written by Claude (AI) for the contributor.*

One `git am` patch for `tools/server/server-context.cpp` (`spec-ckpt-device.patch`), made on r176 (`v15-6050b467c-r176`),
plus the measurement script and its rewrite input.  As with earlier
submissions it is meant to be re-cut into the block that owns the server's speculative path.

## What it does

On a recurrent or hybrid target, the server saves a checkpoint of the target's recurrent state before every verify whose
draft is longer than `n_rs_seq` (`use_ckpt_tgt`), and overwrites it at the next one.  With an n-gram drafter that is any
n-gram draft longer than the rollback window, i.e. most steps of a copy-heavy request.  The checkpoint went to host
memory: for Qwen3.8-27B (150 MiB of recurrent state) a save costs about 16 ms on gfx1201.

The patch passes `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` for these checkpoints, so a save and a restore become device copies
into one buffer per slot that the context keeps and reuses.  Context checkpoints (prompt cache) are not touched.  The
restored state is the same bytes either way, so text and acceptance do not change.

**When device memory is used** (decided once per slot, at its first checkpoint, and logged):

- the target is on exactly one GPU (`split_mode` none, or a single GPU device), and
- that GPU has the state's size plus 512 MiB free at that moment.

Otherwise the slot keeps the host checkpoint as before.  The free-memory check exists because the buffer is allocated
lazily, after the expert-cache arena and the other caches have sized themselves; on a tight GPU it must not take the last
of the memory.  Without the check, a failed allocation inside `llama_io_write_device` aborts the server.

**Kill switch**: `LLAMA_SERVER_SPEC_CKPT_DEVICE=0` keeps host memory always (the release's behaviour).  Per the default-on
policy there is no enabling value.  Proposed `ENVIRONMENT.md` entry:

| variable | default | kind | effect |
|---|---|---|---|
| `LLAMA_SERVER_SPEC_CKPT_DEVICE` | on | kill-switch | `0`: the target's speculative checkpoint (saved before a verify whose draft is longer than `n_rs_seq`) stays in host memory; default: device memory per slot when the target is on one GPU with the state's size + 512 MiB free |

## Not covered (risk)

- **Multi-GPU** (`-sm tensor`, layer split over several GPUs): we have one GPU per machine and could not test it.  The
  patch therefore stays on the host path whenever the target is not on exactly one GPU (the "exactly one GPU" test reads
  `split_mode` / `main_gpu` and the GPU devices in use).  Enabling it there would need a change and a multi-GPU run.
  The device copy path itself (`LLAMA_STATE_SEQ_FLAGS_ON_DEVICE`, `llama_io_write_device`) is upstream's (r157).
- Other architectures than gfx1201.

## Measurements

One R9700 (gfx1201), Linux, ROCm 10.0.0 (TheRock wheels), r176 plus this patch only.  One binary; the arms differ only in
`LLAMA_SERVER_SPEC_CKPT_DEVICE` (`0` = host, the release's behaviour; unset = this patch's default), in the order
0 / unset / unset / 0, one server start per arm, greedy, `ROCBLAS_USE_HIPBLASLT=0` (§44), one request at a time,
`-ub 2048 -b 2048 -fa on -ctk q8_0 -ctv q8_0 -c 32768 --parallel 1 --spec-type ngram-mod,draft-dflash
--spec-draft-n-max 7` (ngram-mod at its default n-max 64).  Script: `specdev176.py` in this directory (its paths are placeholders for the local ones).

- **dense**: Qwen3.8-27B-UD-Q4_K_XL + Qwen3.8-27B-DFlash2-Q4_K_M, `-ngl 99 -ngld 99`.
- **moe**: Qwen3.6-35B-A3B-UD-Q4_K_M + its DFlash Q8_0 drafter, `-ncmoe 40` (all experts host-resident, the expert-cache
  arena active: 30.9 GiB in use after loading).

Requests (all public): `edit` = rewrite a Python file (a 3,000-token prompt; `edit-input.py` here is that file with one
local model path replaced by `models`, which changes the prompt by a token or two) with a comment above each function (`edit-input.py`,
3,072 tokens max; n-gram heavy), `story` = a zh short story, `cont` = an English continuation of the first 24,000 characters (a 5,953-token prompt) of
llama.cpp's own `docs/*.md` concatenated (MIT).  Mean of the two starts per arm, generation t/s:

| surface | request | host (`=0`) | default | change | text, drafts, accepted |
|---|---|---|---|---|---|
| dense | edit | 346.9 | 381.8 | **+10.0 %** | identical |
| dense | story | 29.8 | 29.7 | -0.3 % | identical |
| dense | cont | 73.4 | 73.3 | -0.1 % | identical |
| moe | edit | 413.4 | 478.8 | **+15.8 %** | identical |
| moe | story | 71.2 | 71.1 | -0.1 % | identical |
| moe | cont | 123.9 | 124.2 | +0.2 % | identical |

Each arm's two starts agree within 1.5 % (dense edit 380.6 / 382.9 vs 346.5 / 347.4; moe edit 478.6 / 478.9 vs
410.5 / 416.2).  The gain is where long n-gram drafts make the server checkpoint before most verifies (`edit`: 3,252
drafted / 2,934 accepted tokens on dense); the low-acceptance requests rarely take the checkpoint and do not move.

Decisions logged (one per slot, at its first checkpoint):

- dense: `speculative checkpoints of the target in device memory (149.6 MiB; 11468.0 MiB free on ROCm0)`
- moe: `... in device memory (62.8 MiB; 742.0 MiB free on ROCm0)` / `(62.8 MiB; 724.0 MiB free ...)` -- with the arena in
  place about 0.7 GiB is left, enough for this model's 63 MiB state plus the 512 MiB headroom; a larger state or a fuller
  arena takes the host path.

Device memory in use at the end of the requests: dense 20.79 GiB (host) vs 20.94 GiB (default), +150 MiB = one slot's
state; moe 31.23 vs 31.29 / 31.31 GiB.

## Gates run (r176 + this patch, the same R9700)

- `tests/test-llama-context -m Qwen3.8-27B-UD-Q4_K_XL.gguf -c 8192 --test 'baseline|cp_h|cp_d|cp_h_s|cp_d_s'`: **5 / 5
  passed** (the device io class this patch puts on the server's path, on a hybrid model; the host cases as the control).
- `scripts/gate-ckpt-reuse.sh` with `MODEL` = Qwen3.8-27B-UD-Q4_K_XL, `DRAFT` = DFlash2-Q4_K_M, `SPEC_TYPE=ngram-mod,draft-dflash`,
  the patch on (its default): **PASS** (3 restored checkpoints kept, 5 vs 8 created, all four requests' text identical).
- The host fallback, forced: Qwen3.6-35B-A3B `-ncmoe 40` with `GGML_CUDA_SLAB_HEADROOM_MIB=512`, so the arena leaves less
  than the state plus 512 MiB.  The slot logged `speculative checkpoints of the target in host memory (62.8 MiB; 486.0 MiB
  free on ROCm0)`; text, drafted and accepted tokens identical to `=0`; no abort (401.0 vs 411.9 t/s, one start each, both
  on the host path).
- `scripts/check-gpu-faults.sh` over the kernel journal and every log of these runs and of the measurements: clean.
- Not run: `gate-spec-align.sh` and the other multi-GPU gates (one GPU per machine); `gate-state-pipe.sh` (the host io
  path is not changed -- the patch only stops sending the speculative checkpoint through it when device memory is used).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
