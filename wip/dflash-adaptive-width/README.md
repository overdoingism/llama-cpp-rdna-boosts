# wip/dflash-adaptive-width — DFlash2 adaptive verify width (opt-in), aligned across the sequences drafted together

One `git am` file (one commit), made on `v15-6050b467c-r164` (applied tree `6708bee4`); it applies unchanged on
`v15-6050b467c-r167` (tree `b2150648`), `-r168` (tree `95fe4961`) and `-r170` (tree `bf1d6d08`).  It touches only
`common/speculative.cpp` (+235 / -1), inside the DFlash drafter.
`dflash-adaptive-width.patch` sha256 `911ff798617356feb5104f4f3905bf8524b23fac9d901c3e394b75e6ac5dffe7`.

**This is v3**, after the review of v2 (v2 is the version first submitted in this PR; v1 was an internal draft that was
never submitted):

* **Opt-in.**  `LLAMA_DFLASH_ADAPTIVE=1` turns it on (`2` also logs the width changes and, per request, the verifications
  per width).  Unset or `0` runs the release's code path.
* **Estimates from counts.**  v2 kept `P(position k accepted | k-1 accepted)` as an average that started at 0.7 and moved
  1/32 per draft that reached `k`, so a position the sequence seldom reaches kept an optimistic value and the width
  stayed at `n_max` (the review's trace).  v3 keeps decayed counts per position (hits / checks, decay 31/32 per draft)
  with two pseudo-checks at the previous position's estimate (0.5 for the first).
* **Down two widths at a time, up one, at most once in 8 drafts** (v2: down to any width, once in 16).  A response's
  first drafts often run below its later acceptance; one deep drop on 8 drafts was the costliest mistake we measured
  (prose, 7 -> 3 on the first decision: -4 %).
* **The expert cache's band bound** on the aligned width, as `align_draft_caps()` has it (below).
* **The cost model measured on the release** (below): 0.09 of the step's fixed part per drafted-and-verified row, no
  extra cost at 5 rows.  v2's 0.035 / 0.08 came from our own tree (cheaper verify rows, and a decode band that switches
  tiles at 5 rows); on the release they made widening look cheap.
* *Tried and dropped:* drafting the whole block every time and cutting only what is sent.  The drafted tokens would
  then be the fixed width's and a narrower width's acceptance could be read off each draft.  But on this model most of
  a width's cost is the draft's block, not the verify rows - narrowing only the verify saved 0.35 ms per step on the
  prose prompt, against 2.4-3.1 ms per row of a fixed width - so it lost 3-6 % on prose.  (DFlash2's block attention is
  also not causal, `dflash.attention.causal = false`: a narrower block can draft other tokens, so reading a narrower
  width's acceptance off a wider draft is an approximation, which v3 accepts.)

**Why.** With a fixed `--spec-draft-n-max`, a DFlash2 sequence always drafts and verifies that many tokens.  On text
where the draft is rarely right past the first positions (long Chinese prose in our runs), most of every step goes to
rows that are rejected.

**What it does.**

* *Width.*  Each DFlash2 sequence drafts `w` tokens (`1..--spec-draft-n-max`, starting at `--spec-draft-n-start` or
  `n_max`) and sends them to verification; every 4th draft carries one token past `w`, whose acceptance the target's
  next token tells when the `w` before it are accepted - that is what lets `w` grow.  `w` goes to the width with the most
  expected tokens per unit of step cost, `(1 + sum_k prod_{j<=k} h_j) / (1 + COST * (w + 1))`, only for a gain above 1 %.
* *Estimates from history, not from `accept()`.*  At a sequence's next draft, the previous draft is compared with the
  tokens the target produced after it (`dp.prompt` past the length it had then, plus `dp.id_last`).  A draft token
  equal to the target's token at its position is one the verification accepts - under greedy match and under rejection
  sampling alike.  Nothing overrides `accept()` / `accept_partial()`, so a draft cut after `draft()` returned, a round
  replayed from a checkpoint, or rounds drafted by another implementation cannot skew the width.
* *Alignment (issue #234), hybrid / recurrent targets only, `spec_align_enabled()`.*  The sequences drafted in the
  same `draft()` call send one width: the widest `w` among them, bounded by the lowest hard cap of the call
  (`--spec-draft-n-max` and the caller's per-sequence `n_max`) and by the expert cache's verify band:
  `band = llama_model_moe_cache_band()` (read once the target context exists, 0 without an active arena) bounds it to
  `max(0, band / n_drafting - 1)`, and at 0 each sequence sends its own width - the expressions of
  `align_draft_caps()`.  No cut of finished drafts.  Inert with one drafting sequence.
* DFlash v1, DSpark and LiliCorr drafts, MTP and the n-gram drafters are unchanged.

**Switches.**  `LLAMA_DFLASH_ADAPTIVE` (above); `LLAMA_SPEC_ALIGN=0` drops the alignment (it already gates
`draft-mtp-adaptive`'s); `LLAMA_DFLASH_ADAPTIVE_COST` (default 0.09) and `LLAMA_DFLASH_ADAPTIVE_JUMP` (default 0: the
extra cost from 5 rows, as a fraction of a step at 110K, for a tree whose verify attention switches tiles there).

**Calibrating `COST` on another GPU or model.**  Run the server with `LLAMA_DFLASH_ADAPTIVE` unset at a few fixed
`--spec-draft-n-max` (e.g. 3, 5, 6, 7), take ms per verify step = `predicted_ms / (predicted_n - draft_n_accepted)` from a
greedy request's `timings`, fit `a + b * rows` with `rows = n_max + 1`, and set `COST = b / a`.  Ours, Qwen3.8-27B +
DFlash2 on gfx1201:

| context | a | b per row | b / a |
|---|---|---|---|
| `code-python.txt` (0.5K) | 31.1 ms | 2.41 ms | 0.078 |
| `prose-rdna-boosts.txt` (5.2K) | 30.3 ms | 2.81 ms | 0.093 |
| zh continuation (30K) | 31.3 ms | 3.05 ms | 0.097 |

## Results (v3)

Linux: Ryzen 9 3900X, one R9700 (gfx1201, PCIe 4.0 x16), Ubuntu 24.04.5, ROCm 10.0.0 (TheRock pip), plain release +
this patch; `ROCBLAS_USE_HIPBLASLT=0` in every arm (GREEDY-PURITY §44).  Qwen3.8-27B-UD-Q4_K_XL +
Qwen3.8-27B-DFlash2-Q4_K_M, `-c 245760 -ub 2048 -b 2048 -fa on -ctk q8_0 -ctv q8_0 --spec-type draft-dflash`, greedy, one
server start per arm.  Fixed widths are server starts at `--spec-draft-n-max` 3 / 5 / 6 / 7 (the request-level `n_max`
is disabled in this server); **A** = this patch at `--spec-draft-n-max 7` with `LLAMA_DFLASH_ADAPTIVE=2`, two starts.
The review's three prompts as raw `/completion` prompts, `-n 192`; and a 384-token continuation of a 30K-token Chinese
prompt.

| t/s (vs fixed 7) | fixed 7 | fixed 6 | fixed 5 | fixed 3 | v2 | **A (v3), two starts** |
|---|---|---|---|---|---|---|
| prose | 83.72 | 85.13 (+1.7 %) | 81.61 (-2.5 %) | 73.78 (-11.9 %) | -1.2 % | **82.20 / 81.46 (-1.8 / -2.7 %)** |
| code-python | 101.66 | 102.39 (+0.7 %) | 103.68 (+2.0 %) | 80.18 (-21.1 %) | -1.5 % | **101.63 / 100.84 (0.0 / -0.8 %)** |
| recall | 151.15 | 138.93 (-8.1 %) | 129.82 (-14.1 %) | 95.22 (-37.0 %) | -1.2 % | **151.21 / 150.16 (0.0 / -0.7 %)** |
| zh 30K | 30.04 | 32.29 (+7.5 %) | 34.44 (+14.6 %) | 38.21 (+27.2 %) | +13.5 % | **35.67 / 35.58 (+18.7 / +18.4 %)** |

Verifications per width in A (both starts the same): prose `5:9 6:10 7:24`, code-python `7:34`, recall `7:23`, zh
`1:45 2:148 3:29 5:8 7:8`.  Our `code-python.txt` run accepts far more than the review's (155 of 192 tokens at width 7):
we send the file as a raw prompt, which may not be how the review ran it, so its mid-acceptance code case is not
reproduced here.

**Text.**  Every request in every arm - fixed 3 / 5 / 6 / 7, v2, A in both starts - produced the same text as fixed 7,
including the requests whose width moved; and the release's path in two builds (this one with the variable unset, the
v2 build with `=0`) matched it byte for byte.  Per §63 a verify width is a numeric axis, so a moved width is not promised
to keep the text; it did on these four requests.

**Two slots** (`--parallel 2 --kv-unified`, code-python and recall sent at once, twice per start): aggregate 82.5 /
122.2 t/s with fixed 7 and 82.6 / 126.1 t/s with A (the first pair includes both prompts' prefill); the pair texts were
the same in both arms.  The alignment's measurements from v2 below still describe it (only the band bound is new).

## Earlier measurements (v2, default-on, same alignment)

`--spec-draft-n-max 5`.  Two sequences at once on Linux: slot 0 a zh continuation (low acceptance), slot 1 an en
continuation (high acceptance), 384 tokens each; ms per step while both slots verify, means of two starts:

| ms per step, both slots verifying | release | adaptive, not aligned | adaptive, aligned |
|---|---|---|---|
| 8K each | 65.3 | 69.1 (+5.8 %) | 64.2 |
| 32K each | 92.0 | 92.3 | 92.1 |
| 50K each | 112.3 | **108.5 (-3.4 %)** | 112.0 |

Aligned, the patch cost what the release costs at every length; unaligned, the split cost 6 % at 8K, nothing at 32K
and was 3 % cheaper at 50K (two slots at width 5 make a 12-row batch, past the <= 8-row decode/verify band, while the
split ubatches stay inside it; not profiled).  The alignment stays unconditional, as in `align_draft_caps()`.

What moved the text in those runs, besides a moved width: the process start without the hipBLASLt pin (§44); a zh
100K continuation that still took one of two texts per start with the pin (both arms sharing one of them); a second
slot generating at the same time (another numeric axis of the same kind); and the arrival points of a pair's two
requests, which batch the first steps differently from start to start - which is why the two-slot comparison used step
time, not text.  On Windows (Ryzen 7 5700G, one R9700) every request reproduced in every start without any pin.

**Not covered:** `-sm tensor` and `-ncmoe` with the expert-cache arena - we have one GPU, and the only MoE draft we have
(for a 35B-A3B) is DFlash v1, which this patch leaves alone, so the band bound ran inert in every arm above (dense 27B,
band 0); with the patch opt-in, nothing changes on those configs unless it is enabled.  GPUs other than gfx1201 (see
the calibration above).  v3 on Windows.  The n-gram-first chain (`ngram-mod,draft-dflash`).  Sampling at temperature >
0 for v3.

*Written by Claude, an AI model by Anthropic, from measurements on the contributor's machines.*
