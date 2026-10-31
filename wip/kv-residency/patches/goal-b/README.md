# `patches/goal-b` — the Goal B series (B-A + the two delivery fixes), self-contained

**Not delivery work** (same rule as the parent directory): nothing here is in `patches/`,
`release.json` or `rdna-boosts-all.patch`, and nothing may be applied to the fork's `rdna-boosts`
branch without the promotion go-ahead.  Stored so a cold session can reproduce the exact tree the
campaign's Goal B numbers were measured on, without depending on the fork branch existing.

## How to reproduce the tree

The series applies to the **r186** release tip, `8f9dad0be` (`v16-6050b467c-r186`; tree
`0bd26dbe195185072019fe99926177d4daa7718b`, which is `release.json`'s tree at r186).  Re-cut it with
`scripts/apply-all.sh` from an older checkout if needed (`RDNA_BRANCH=rebase-r186`).

```bash
cd ~/llama.cpp && git checkout -b wip-kv-residency-b rebase-r186
git am ~/llama-cpp-rdna-boosts/wip/kv-residency/patches/goal-b/00*.patch     # 0001..0007, in order
~/bin/build-llama-rocm
```

**Verified 2026-10-31:** `git am` 7/7 clean, applied tree **`14e76aa2af4135b13cd3657895f04460d3d80774`**,
which equals the campaign branch tip `wip-kv-residency-b` (`b98cb3e43`).  `SHA256SUMS` has the
per-patch digests; `diffstat.txt` is the net (22 files, +1360/−69).

## The series

| # | commit | what |
|---|---|---|
| 0001 | `b86b47e51` | **B1**: per-cache window override — DIAGNOSTIC, campaign only (lets one of Flash-Next's two KV caches take the block-15 window) |
| 0002 | `7a4a03571` | **B2**: a *streaming* KV takes the sparse QSA decode arm (`qwen4exp_kv_streams`, `LLAMA_QSA_DENSE_DECODE_STREAM=0` = A/B) |
| 0003 | `8725d885a` | **B1(c)**: attended-set census (top-k trace + `qsa-census.py` replayer) — diagnostic |
| 0004 | `7649f4898` | **B-A**: `LLAMA_KV_REGION_POOL` — pinned host mirror + compact device pool + page table + `GGML_OP_KV_POOL` resolve; a second id list for the QSA reader |
| 0005 | `922604069` | **B-A**: the pool mirror is host memory — accept it in the async state copies (the ckpt/state gates) |
| 0006 | `f8e47e392` | **A12 — a DELIVERY fix, not campaign code**: `cpy_2d_vec16`, a vectorised fallback for the 2-D copy r183's guard refuses (block 10 candidate; also staged as `beta/0016-copy-2d-vec16/`) |
| 0007 | `b98cb3e43` | **A13 — a DELIVERY fix**: the fused indexer scorer accepts quantized keys (q8_0 + the q4/q5 family + iq4_nl), byte-identical to the per-op chain on all eight types (block 11 candidate) |

Numbers, gates and the reasoning: `../BA-POOL-DESIGN.md` §7 (B-A), §8 (the acceptance table), §9
(A12 + the root cause), §10 (A13).  The next session's list: `../COLD-START-PROMPT.md`.

**0006 and 0007 are delivery fixes in their own right** — they repair a shipped regression (r183's
copy fallback) and a shipped performance gate (quantized indexer keys skipped the fused scorer) and
do not depend on the pool being promoted.  If the pool is dropped they still land (blocks 10/11);
if they land first, this series' remaining five patches are the pool itself.
