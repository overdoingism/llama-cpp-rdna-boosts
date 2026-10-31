# Cold-session prompt — `wip/kv-residency` (TODO #118)

Paste the block below into a fresh session.  It assumes only what is in the repo (nothing from the previous
session's context).  **Rewritten 2026-10-31, end of session A13.**

---

Continue the windowed-KV campaign (TODO **#118**) in `~/llama-cpp-rdna-boosts` and `~/llama.cpp`.

## This session's goal: FINALISE AND DELIVER Goal B — do not re-measure it

**The maintainer's decision (recorded 2026-10-31): Goal B is to be wrapped up and delivered into the delivery
repo.**  The plan is [`HANDOVER.md`](HANDOVER.md) **§12 — read it first, then work down its 12.1 table**; do not
re-derive it.  Goal C and the open Goal A items are carried forward (see the end of this prompt), not dropped.

**Start by** reading, in this order: `HANDOVER.md` **§12** (the close-out plan), then
[`BA-POOL-DESIGN.md`](BA-POOL-DESIGN.md) **§7-§10** (what the pool is, every gate, the numbers, the two repairs),
then `HANDOVER.md` §3 (invariants), §5 (measurement rules — **rule 9** is the `-n 512`/acceptance rule), §6 (gates).

### What exists (do not rebuild, do not re-measure)

* **The pool** — `LLAMA_KV_REGION_POOL=<cells>` (unset/`0` = off, byte-identical): the attention KV becomes a
  pinned host **MIRROR** (authoritative, so `seq_rm`/`seq_cp`/defrag/state/park run unchanged) plus a compact
  **device pool** of indexer-block pages (4 cells), a page table `tbl[block] -> slot`, and **`GGML_OP_KV_POOL`**
  (mark / claim / map+fill, 3 launches per layer per step) resolving inside the captured graph; the QSA reader
  takes a second id list so its K/V gather follows the table while the mask stays cell-indexed.
* **Gated and measured**: text identical pool on/off at 5.3k (`296 91ef8940d66e` / `332 dd1981213373`) and at
  51 475 depth (`854 43b30de151b9`); `LLAMA_KV_POOL_VERIFY` 0 mismatches; prefill-logit 0.000000 / 100.000 %;
  park / ckpt / state gates PASS; `FLASH_ATTN_QSA` 27/27; portability lint clean on the two files it adds.
  Acceptance (`-n 512`, MTP depth 4, arms interleaved): **5.3k 72.09 / 71.58 / 71.27**, **103k 59.10 / 54.70 /
  54.65** (dense / sparse / pool), 180k 48.73 / 42.93 / 43.09 (**stale — task 0**).  Arena gain +1.44 GiB, expert
  hit 96.26 → 97.02 %.  The pool lands **on the sparse ceiling at every depth**; the pool's own cost is
  0.10-0.11 ms/token with fills 0.03-0.05 ms/step, which is why the page policy is **dead** (do not touch it).
* **Two repairs, already built and gated, that are DELIVERY fixes rather than campaign code** — they fix a shipped
  regression and a shipped performance gate, and they do not depend on the pool:
  * **A12** `f8e47e392` → **block 10**: `cpy_2d_vec16`, a vectorised fallback for the 2-D copy r183's guard refuses
    (qwen4exp + quantized KV + the sparse arm paid 5.20 ms/token in an element-per-thread copy kernel).
  * **A13** `b98cb3e43` → **block 11**: the fused `INDEXER_SCORE`/`INDEXER_FILL` path now accepts **quantized**
    indexer keys (q8_0 + the q4/q5 family + iq4_nl), byte-identical to the per-op chain on all eight key types;
    **plain decode 23.68 → 17.73 ms/token and 29.3 → 36.3 t/s** at 103k.
* **Where the code lives** (all pushed): fork branch **`wip-kv-residency-b`** (tip `b98cb3e43`, based on the r186
  delivery tip `8f9dad0be`; the pre-rebase tip is branch `wip-kv-residency-b-r182` = `b5d895c59`);
  self-contained patch series **`wip/kv-residency/patches/goal-b/0001..0007`** (`git am` reproduces the branch
  tree `14e76aa2a…` exactly, verified); and A12 staged as a beta candidate —
  **`beta/0016-copy-2d-vec16/`** (patch cut against r188, `git am`-verified there, builds, text identical).

### The work, in order (details and acceptance criteria: `HANDOVER.md` §12.1-§12.6)

0. **Measure the one stale row**: `wip/kv-residency/acceptance-arms.sh 180k` (one interleaved batch, `-n 512`,
   MTP depth 4) and update `BA-POOL-DESIGN.md` §8.2.  That is the only measurement left.
1. **Re-base** the campaign branch onto the *current* `release.json.tip` (re-cut it with
   `RDNA_BRANCH=rebase-r<NN> scripts/apply-all.sh .`; never a hard-coded SHA), keep the old tip as a
   `*-r<NN>` branch, push both, and re-run the gate list on the new base (12.1 row 1).
2. **Ship the two repairs first** — fold A12 into **block 10** and A13 into **block 11** and cut the set, **or**
   ship them as their own micro-release ahead of the pool (the lower-risk option; the maintainer's call).
3. **`--fit` accounting for the pool — the hard blocker** (§12.2): `common/fit.cpp` recognises only the windowed-KV
   buft (`fit_buft_is_kv_region`, ~line 138), so the pool's mirror bytes are booked as **VRAM** and `--fit` reserves
   1.6-2.4 GiB for host memory — i.e. it gives back the arena the feature exists to free.  Add a pool-buft predicate
   next to `ggml_backend_cuda_kv_pool_mirror_buft`, accept it in the fit, and carry out `min(mirror, pool cells ×
   cell bytes)`.  **If this cannot be made to read correctly, do not ship the pool** (rollback criterion).
4. **Decide the grow-on-demand mirror** (§12.3) — implement it, or ship it as a documented limitation (the whole
   `-c` is pinned at creation: 0.2 / 1.6 / 2.4 GiB at 16k / 131k / 196k).  Record the decision; if deferred, add it
   to `TODO.md` as an open follow-up.
5. **Package the pool as block 16** (§12.4): `patches/0016-*`, `release.json` via `scripts/make-release.sh`,
   `patches/README.md` block note, `ENVIRONMENT.md` § (`LLAMA_KV_REGION_POOL`, `LLAMA_KV_POOL_VERIFY`,
   `LLAMA_KV_POOL_STATS`), `MANIFESTS.md` apply order + verification contract, `AGENTS.md` naming (17 blocks) and the
   default-off rationale, `WORKLOG.md` entry, and close `TODO.md` #118 to it.  **Default OFF** — it is a fittability
   feature like block 15 (on a config whose KV fits it costs the sparse-vs-dense gap).
6. **Release**: `scripts/validate-set.sh` green → tag → `scripts/fork-refresh.sh --build --gates --push` → the
   container watcher on `radiance` publishes the images (never build them on the primary box).
7. **Refresh the campaign record**: re-export `patches/goal-b/` from the final branch (`SHA256SUMS`,
   `diffstat.txt`), and shrink this prompt to what is left (Goal C + the Goal A `auto` decision).  Do **not** archive
   `wip/kv-residency/` while Goal C is open.

### Rules that cost a session each (short list)

* **Every Flash-Next arm** carries `-ot per_layer_token_embd.*=CPU -lm none -lzm off` (otherwise ±20 %), and needs
  the separate MTP head `-md …/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`.
* **`-n 512` and matched tokens/step** before reading any t/s delta (an MTP arm's t/s is tokens/step × step rate;
  the old "−20 %" was a `-n 128` window artefact).  Static MTP depth ≤ 4.
* **`ROCBLAS_USE_HIPBLASLT=0`** for every fresh-process numeric comparison — the maintainer's `mmf32` bypass covers
  the **F32** prefill shapes only; Flash-Next's **BF16** prefill still goes through hipBLASLt's per-start solution
  lottery (issue #67 / `rocm-libraries#12126`).
* **`/tmp` is a 47 GB tmpfs** (it is RAM — the reason the box nearly starved): move a prefill-logit base to
  `/var/tmp/kv-gate/`, delete rocprof `*_kernel_trace.csv` after analysis (only `*_kernel_stats.csv` matters).
* One GPU job at a time; `rocm-smi --showmeminfo vram` before any gate (the maintainer's server holds ~34 GB);
  always `--single-turn --no-display-prompt`; always `timeout`; text from runs **without** `-v`
  (`scripts/extract-generated.py`), timings from a separate `-v` run.
* **Push after every step** — both the delivery repo and the fork branch.  Never push to upstream llama.cpp;
  the only fork targets are `stew675/llama-cpp-rdna-boosts` and `git@github.com:stew675/llama.cpp.git`.
* Findings go to the maintainer **before** a delivery-affecting commit (AGENTS.md), but the promotion decision for
  Goal B is already made — do not stop to re-ask.

### Carried forward (not this delivery, keep them alive)

* **Goal C — the drafter ring** (`O3-DRAFTER-RING.md`): measured, **not** promoted (free at a usable window, 7-9 %
  lost far below it, and the VRAM prize does not reach the arena).  Three questions for the maintainer before any
  promotion: default-on or opt-in, one window size or per-model, MTP-only or DFlash2 too.
* **Goal A `auto` policy** (`A2-MIXED-BACKING.md` §3.7.11.4 A-C) — measured; the maintainer's call.
* **Goal A across `-sm tensor`** (`A6-TENSOR-SPLIT-WINDOW.md`) — designed only, hardware-blocked here.
* **The radix top-k** (2.03 ms/token plain, 0.9 ms/step under MTP) — the largest sparse-arm item left after A13.

### Do not redo

Goal A; B1/B2/B1(c); B-A's design, implementation or gates; the acceptance table's method or the census; the
"is the pool exact?" question; the page policy (measured dead); the MTP draft's sparse-decode re-measure
(default-OFF holds); the A12/A13 fixes; the rebase-and-regression hunt.  All of it is in
`BA-POOL-DESIGN.md` §7-§10 with the numbers and the commands.
