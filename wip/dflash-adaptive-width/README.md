# wip/dflash-adaptive-width — DFlash2 adaptive verify width, aligned across the sequences drafted together

One `git am` file (one commit), made on `v15-6050b467c-r164` (applied tree `3d03d7fd`); it applies unchanged on
`v15-6050b467c-r167` (`git am`, tree `fc8d714e`; r164 -> r167 did not touch `common/speculative.cpp`, `tools/server/`
or the batch splitter).  It touches only `common/speculative.cpp` (+213 / -1), inside the DFlash drafter.
`dflash-adaptive-width.patch` sha256 `db346deb05e6fe185ffd6ff40f28aa932f324a617289d1cad98e7d3a0f4ecdf1`.

**Why.** With `--spec-draft-n-max 5` a DFlash2 sequence always sends 5 drafted tokens to verification.  On text
where the draft is rarely right past the second or third position (long Chinese prose in our runs: about 7 % of
the drafted tokens accepted at width 5), most of every verify step is spent on rows that are rejected, and on gfx1201 the decode
attention band leaves its 4-row tile at 5 rows, which makes the 5th and 6th row expensive at long context.  A
width per sequence that follows what the draft actually achieves recovers that time; the output tokens are still
the target's own greedy samples.

**What changes.**

* *Width.*  Each DFlash2 sequence drafts `w` tokens (`1..--spec-draft-n-max`, starting at `--spec-draft-n-start`
  or `n_max`).  `w` follows running per-position estimates of P(position k accepted | k-1 accepted) to the width
  with the most expected tokens per unit of step cost.  Cost model: a verified row costs 0.035 of a step, and from
  5 rows on a step costs 0.08 more at 110K positions (scaled linearly with the context, at most twice) - both
  measured on gfx1201 on our own tree, which carries this band (`+3.9 ms` of a 49 ms step at 110K, next to nothing
  at 8K); not re-derived on the release.  The width
  may drop to any value or rise by one, at most once in 16 drafts and only for a gain above 1 %, so the target
  rebuilds its verify graph rarely.  Every 4th draft carries one token past `w`, so the estimate for the next
  position stays fresh and `w` can grow back.
* *Estimates from history, not from `accept()`.*  At a sequence's next draft, the previous draft is compared with
  the tokens the target produced after it (`dp.prompt` past the length it had then, plus `dp.id_last`).  A draft
  token equal to the target's token at its position is exactly one the verification accepts - under greedy match,
  and under rejection sampling too, where a rejected token is never the replacement.  So nothing overrides
  `accept()` / `accept_partial()`, and a draft cut after `draft()` returned, a round replayed from a checkpoint, or
  rounds drafted by another implementation in between cannot skew the width.
* *Alignment (issue #234), hybrid / recurrent targets only, `spec_align_enabled()`.*  The sequences drafted in the
  same `draft()` call send one width: the widest `w` among them, bounded by the lowest hard cap of the call
  (`--spec-draft-n-max` and the caller's per-sequence `n_max`, which the server lowers as a response runs out of
  tokens - the same bound `align_draft_caps()` uses for `draft-mtp-adaptive`).  Without it two slots at different
  widths verify in two ubatches (`split_equal()`, GREEDY-PURITY §63).  There is no cut of finished drafts.  With a
  single drafting sequence it is inert.  Aligned or not, the width respects the per-sequence `n_max`, so the
  wrapper's post-draft truncation never applies to these drafts.
* DFlash v1, DSpark and LiliCorr drafts, MTP and the n-gram drafters are unchanged.

**Kill switches.**  `LLAMA_DFLASH_ADAPTIVE=0` keeps the width fixed at `--spec-draft-n-max` (the release's
behaviour; `=2` also logs width changes and, per request, the verifications per width).  `LLAMA_SPEC_ALIGN=0` drops
the alignment (it already gates `draft-mtp-adaptive`'s).  `LLAMA_DFLASH_ADAPTIVE_COST` / `LLAMA_DFLASH_ADAPTIVE_JUMP`
override the two cost constants (`JUMP=0`: no step at 5 rows).

## Results

Plain release + this patch, nothing else.  Qwen3.8-27B-UD-Q4_K_XL + Qwen3.8-27B-DFlash2-Q4_K_M, `-ngl 99 -ngld 99
-ub 2048 -b 2048 -fa on -ctk q8_0 -ctv q8_0 --spec-type draft-dflash --spec-draft-n-max 5`, greedy unless noted.
Arms from one build: **F** = `LLAMA_DFLASH_ADAPTIVE=0` (= the release), **A** = this patch's default, **U** = adaptive
with `LLAMA_SPEC_ALIGN=0`.  Each arm twice per machine, interleaved (F U A A U F / F A A F), one server start each.

* **Linux**: Ryzen 9 3900X, one R9700 (gfx1201, PCIe 4.0 x16), Ubuntu 24.04.5, kernel 7.0.0-38, ROCm 10.0.0 (TheRock
  pip, `libamdhip64` 7.15.26333), `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DGGML_HIP_NO_VMM=ON`; `-c 245760
  --parallel 2 --kv-unified`; **`ROCBLAS_USE_HIPBLASLT=0` in every arm** (see "Reproducibility").
* **Windows 11**: Ryzen 7 5700G, one R9700, TheRock ROCm 10, driver 32.0.32015.7029; `-c 131072 --parallel 1`.

### One sequence

| | F | A | |
|---|---|---|---|
| Linux, zh continuation of a 29.9K-token prompt, 384 tokens | 27.28 t/s, 49.22 ms/step | 29.14 t/s, 46.07 ms/step | **+6.8 %**, same text in every start |
| same, temperature 0.7 (an earlier run without the BLAS pin) | 28.32 t/s | 29.79 t/s | +5.2 %, same text |
| Linux, zh continuation of a 100.6K-token prompt, 384 tokens (the one text both arms produced) | 29.11 t/s, 55.89 ms/step | 31.27 t/s, 50.33 ms/step | **+7.4 %** |
| Windows, zh short story (t 0.3, seed 1, 1536 tokens) | 43.7 t/s | 45.6 t/s | **+4.3 %**, same text |
| Windows, zh 8K continuation, 384 tokens | 29.4 t/s | 30.5 t/s | **+3.8 %**, same text |
| Windows, whole-file code rewrite, three ERP tool-call turns, a thinking question, en 8K / 30K / 100K continuations | | | **-0.4 .. +0.1 %**, same text; the width stays at 5 |

(Windows: against F's second start; its first start was slower on every request, a warm-up effect.)  The width the
model settles on for the zh text is 3 (verifications per width on the 100K prompt: `2:0 3:216 4:0 5:26`); the en
continuations and the code rewrite stay at 5.

### Two sequences at once (Linux)

Slot 0 a zh continuation (low acceptance), slot 1 an en continuation (high acceptance), both from the start of their
texts, primed, then 384 tokens each sent at once.  The streamed token times are grouped into verify steps (a step's
tokens arrive within a few ms, steps are 60+ ms apart), which gives the step time while both slots emit - the
quantity the alignment acts on, and one that does not move with which continuation a near-tie picked (the text of
a concurrent pair is not stable from start to start, see below).  Means of two starts:

| ms per step, both slots verifying | F (release) | U (adaptive, not aligned) | A (adaptive, aligned) |
|---|---|---|---|
| 8K each | 65.3 | 69.1 (+5.8 %) | 64.2 |
| 32K each | 92.0 | 92.3 | 92.1 |
| 50K each | 112.3 | **108.5 (-3.4 %)** | 112.0 |

At 8K, in starts where the zh slot took the same continuation (1.35 tokens per step) the pair ran 75.7 t/s with A,
70.9 with U (+6.8 % for the alignment) and 75.3 with F.

**The unaligned split is not uniformly worse.**  Aligned, the patch costs what the release costs at every length
we ran, which is what the alignment is for.  Unaligned, the split costs 6 % per step at 8K, nothing at 32K, and is
3 % *cheaper* at 50K.  Our reading, not profiled on this tree: two slots at width 5 make a 12-row verify batch,
past the <= 8-row decode/verify band, while the split ubatches (e.g. 6 rows, then 3) each stay inside it; at 8K the
second pass over the weights dominates, at 50K the attention does.  We kept the alignment unconditional, as
`align_draft_caps()` does: it is never worse than the release, and a context-dependent rule would need the batch
split's cost in the model.  On our own tree, where 9-16-row verify batches run as two halves through the band, the
unaligned adaptive width cost the pair 16 % (8K) and 23 % (50K) of its throughput, so the size of this trade
depends on how the tree handles verify batches above 8 rows.

## Greedy text

Per GREEDY-PURITY §63 a verify width is a numeric axis, so the adaptive width is a new per-config reference and is
not promised to reproduce the fixed width's text; every emitted token is still the target's greedy sample at its
position.  What actually moved the text in our runs, and what did not:

* **Did not move it: the adaptive width on one sequence.**  Every one-sequence request produced the same text with
  F and A wherever the runs were reproducible at all: all ten Windows requests (including the two whose width
  dropped to 3), the Linux 30K continuation greedy and at temperature 0.7 (seeded), and the Linux 100K continuation
  in the one start per arm that landed on the same text.  We saw no width-caused difference, but §63's caveat
  stands.
* **Moved it: the process start (Linux).**  Without `ROCBLAS_USE_HIPBLASLT=0` the same arm, same request, gave
  different text in different starts (zh 50K: two continuations, 26 vs 82 t/s; zh 100K: three different texts in
  four starts).  With the pin the 30K and 8K requests reproduced, but the zh 100K continuation still took one of
  two texts per start - with both arms sharing one of them, so not the width (see Reproducibility).
* **Moved it: a second slot generating at the same time.**  The same prompt generated alone and generated while
  the other slot was verifying gave different text in 11 of the 12 pairs of our first run, F (the fixed width)
  included: the verify batch holds the other slot's rows, another numeric axis of the same kind.
* **Moved it: when the two requests of a pair arrive.**  The two requests never reach the server at exactly the
  same point, so the first steps of a pair batch differently from start to start: with the pin, the 32K and 50K
  pair texts differed between the two starts of every arm, and so did F's 8K pair (U's and A's 8K pairs
  repeated).  This is why the two-slot comparison uses step time, not text or t/s.
* **Moved it by construction: the alignment with two slots.**  It changes a slot's verify width, so U and A pair
  texts differ (§63's "two configs").

## Reproducibility

* Linux: any cross-start text comparison needs `ROCBLAS_USE_HIPBLASLT=0` (§44); every Linux number above was taken
  with it, except the temperature-0.7 row (whose text happened to match in every start).
* With the pin, the zh 100K continuation still split into two texts across starts, independently of the width.  A
  second per-start source on this model at this length, beyond the hipBLASLt pick; we did not chase it.
* Windows: every request reproduced in every start without any pin.

**Not covered:** multi-GPU (`-sm tensor`), the n-gram-first chain (`ngram-mod,draft-dflash`: there the mismatch is
the n-gram draft's content against the DFlash block, which this patch does not touch - no cut is reintroduced),
non-hybrid targets (no alignment there), GPUs other than gfx1201 (the two cost constants come from this band; the
override variables exist for that), `--spec-draft-sampling probabilistic` at temperature > 0 beyond the one 0.7 row.

*Written by Claude, an AI model by Anthropic, from measurements on the contributor's machines.*
