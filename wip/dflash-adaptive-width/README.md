# wip/dflash-adaptive-width — DFlash2 adaptive verify width (opt-in), aligned across the sequences drafted together

One `git am` file (one commit), made on `v15-6050b467c-r171` (applied tree `855a02d3`); it applies unchanged on
`v15-6050b467c-r172` (tree `6e3f6e65`) and `-r170` (tree `933621b3`).  It touches only `common/speculative.cpp`
(+335 / -1), inside the DFlash drafter.  `dflash-adaptive-width.patch` sha256
`6d8bec95d6d3dcdd5c043cf2139f0d9dd26274748ac3348418939ae2ed3326b3`.

**This is v5.**  v2 is the version first submitted in this PR, v3 the one after its first review.  v4 (the second
review's points, with the cost still a constant) was not pushed: a wider test showed it 3-6 % slower on two shapes (an
8K English continuation, and our production chain at temperature 0.7, where a verified row saved half of what the
greedy calibration said), i.e. the same failure as the review's arena arm, where 0.09 argued for width 3 and the
measurements said 7.  v5 replaces the constant with the measured step time.

## What changed since v3 (the review's list, in its order)

1. **The width carries into the next turn of a conversation.**  A request whose prompt continues the sequence's last one
   (the last prompt's 32 tokens ending 16 before its end appear at the same place; the 16 skip the generation prompt a
   chat template may render differently in the history) keeps the width; the estimates restart.  Any other request starts
   again from `n_max`.  v4 kept the width for every request; a simulation showed a slot that had gone down to 2 on
   low-acceptance text losing 22 % on the next, unrelated, high-acceptance request.
2. **No move down on unobserved positions.**  A narrower width has to win even with every position checked in fewer than
   2 (decayed) drafts taken as always accepted, so the `0.0/0.0` rows of the review's trace cannot argue for it.  A move up
   needs the position past `w` observed (every 4th draft carries one token past `w`).
3. **One width per move, and a window before each.**  A request's first move down waits for 24 drafts, every later move for
   8 drafts at the new width.  A move down needs an estimated gain above 3 %, a move up above 1 %.
4. **The step time is measured.**  For each width the request has run at: the time from the start of one draft of the
   sequence to the start of its next (its draft, the verification and whatever else the step held), averaged over the
   width's last 8 steps; a step more than 4x the width's average is left out as holding other work.  A width with 4 such
   steps is costed by its own time.  A width not yet run at is costed by the model (`1 + COST * max(w + 1, FLOOR)`, COST
   0.09 or 0.15 with an active arena, FLOOR 4), scaled to the measured width nearest to it.  So the model only proposes
   a move and the measurement at the new width decides whether it stays: a move whose saving falls short is undone at the
   next decision.  Widths are compared by their own measured times, so no slope has to be fitted from a sequence that
   stays at one width (the reason we had first held back from this).
5. **Not done: warm-up self-calibration.**  With 4 in place, the model's error costs at most one window at a wrong width.

`FLOOR`: on gfx1201 a step at width 1-2 was no cheaper than one at width 3 (Qwen3.8-27B + DFlash2 at 30K: 46 ms against
44 ms), so the model costs fewer than 4 rows as 4 and the width does not go below 3 on the model's word.

## What it does

* *Width.*  Each DFlash2 sequence drafts `w` tokens (`1..--spec-draft-n-max`, starting at `--spec-draft-n-start` or
  `n_max`) and sends them to verification.  `w` moves towards the width with the most expected tokens per unit of step
  time, `(1 + sum_k prod_{j<=k} h_j) / step(w)`, under the rules above.
* *Estimates from history, not from `accept()`.*  At a sequence's next draft, the previous draft is compared with the
  tokens the target produced after it (`dp.prompt` past the length it had then, plus `dp.id_last`).  A draft token equal to
  the target's token at its position is one the verification accepts, under greedy match and under rejection sampling
  alike.  Per position the estimate is a decayed count (hits / checks, decay 31/32 per draft) with two pseudo-checks at the
  previous position's estimate.  Nothing overrides `accept()` / `accept_partial()`.
* *Alignment (issue #234), hybrid / recurrent targets only, `spec_align_enabled()`.*  The sequences drafted in the same
  `draft()` call send one width: the widest `w` among them, bounded by the lowest hard cap of the call and by the expert
  cache's verify band (`max(0, band / n_drafting - 1)`; at 0 each sequence sends its own width), as `align_draft_caps()`
  does.  No cut of finished drafts.  Inert with one drafting sequence.  Unchanged since v3.
* DFlash v1, DSpark and LiliCorr drafts, MTP and the n-gram drafters are unchanged.

**Switches.**  `LLAMA_DFLASH_ADAPTIVE`: unset or `0` runs the release's code path (every new line is behind `if (adapt)`);
`1` turns this on; `2` also logs each width change with the per-position counts, whether a request kept the width, and
per request the verifications per width.  `LLAMA_SPEC_ALIGN=0` drops the alignment.  `LLAMA_DFLASH_ADAPTIVE_COST` (default
0.09, 0.15 with an active arena), `LLAMA_DFLASH_ADAPTIVE_FLOOR` (default 4) and `LLAMA_DFLASH_ADAPTIVE_JUMP` (default 0)
override the model, which now only decides where to try first.

## Results (v5)

R is the same binary with `LLAMA_DFLASH_ADAPTIVE` unset, v5 is `=2`, v2 is the version first submitted (on by default),
built on the same tree.  One server start per arm, one request at a time, `ROCBLAS_USE_HIPBLASLT=0` in every arm (§44).
Each figure is t/s against the mean of R's starts in the same run.

### Linux

Ryzen 9 3900X, one R9700 (gfx1201, PCIe 4.0 x16), Ubuntu 24.04.5, ROCm 10.0.0 (TheRock pip), plain `v15-6050b467c-r171`
+ this patch.  Qwen3.8-27B-UD-Q4_K_XL + Qwen3.8-27B-DFlash2-Q4_K_M, `-c 98304 -ub 2048 -b 2048 -fa on -ctk q8_0 -ctv q8_0
--kv-unified --parallel 2 --jinja`.  Two runs: R v5 R v5, then R v5 v2 v5 v2 R.

*Config A*, `--spec-type draft-dflash --spec-draft-n-max 7`, greedy.  Slot 0: a Thai continuation, then an 8K-token English
continuation (raw `/completion`, 256 tokens each).  Slot 1: rewriting a whole 1K-token Python file with a comment above
each function (chat, thinking off, 1024 tokens).

*Config B*, our production chain: `--spec-type ngram-mod,draft-dflash --spec-ngram-mod-n-max 15 --spec-ngram-mod-n-min 8
--spec-draft-n-max 5 --spec-draft-sampling probabilistic`.  Slot 0: one Chinese short-story prompt (chat, thinking on,
1024 tokens) at temperature 0.7 with seed 1, seed 2, then greedy; the same prompt again counts as a continuing request,
so the second and third start from the width the one before left.  Slot 1: the rewrite, greedy.

| | R t/s | v5, run 1 (two starts) | v5, run 2 (two starts) | v2, run 2 | v5 widths |
|---|---|---|---|---|---|
| A Thai | 36.1 | **+14.9 / +14.8 %** | **+15.2 / +14.8 %** | +9.4 % | 7 -> 3 from the 24th draft (82 of 131 verifications at 3) |
| A English 8K | 91.4 | -0.7 / -0.5 % | -0.7 / -1.0 % | -1.9 % | stayed at 7; drafts as R's |
| A rewrite | 95.0 | -1.9 / -1.6 % | -1.1 / -1.3 % | -0.2 % | 7 -> 6 -> 5 -> 6 -> 7 |
| B story, T 0.7 seed 1 | 41.1 | -2.6 / +8.7 % | +1.5 / -2.4 % | +4.0 % | 5 -> 3/4 |
| B story, T 0.7 seed 2 | 36.6 | +8.7 / +6.6 % | +8.8 / +8.9 % | +4.8 % | mostly 4 |
| B story, greedy | 37.6 | +7.7 / +8.2 % | +8.2 / +8.1 % | +2.3 % | 4, one window at 3 |
| B rewrite | 206.5 / 209.4 | -4.6 / -0.3 % | +0.1 / +0.1 % | 0.0 % | stayed at 5; drafts as R's |

The -4.6 % start of the B rewrite never moved the width and drafted exactly what R drafted (947 accepted of 978); the
other three v5 starts were -0.3 to +0.1 %.  The A rewrite is the cost of a move that did not pay: down on its early drafts,
back up once the measured times said so.

### Windows

Ryzen 7 5700G, one R9700, Windows, TheRock ROCm 10.0.0, plain `v15-6050b467c-r171` + this patch.  `-ub 2048 -b 2048 -fa on
-ctk q8_0 -ctv q8_0 --kv-unified --parallel 2 --jinja`, `/v1/chat/completions`.  Arms R v5 v2 v5 R v2 per block.

* **qA**: Qwen3.8-27B-Uncensored-Q5_K_M (a fine-tune of Qwen3.8-27B) + Qwen3.8-27B-DFlash2-Q4_K_M, `-c 65536 --spec-type
  draft-dflash --spec-draft-n-max 7`, greedy, thinking off.  One-turn prompts one after another on slot 0 (each another
  conversation), a 3-turn English role play on slot 1.
* **qB**: the same model in config B's chain (`ngram-mod,draft-dflash` 15/8, `--spec-draft-n-max 5`, probabilistic),
  `-c 65536`, at temperature 0 and 0.7 (seed 1).
* **mA**: another family, Muse-Glimmer-30B-UD-Q5_K_M + z-lab's Muse-Glimmer-30B-DFlash2-Q4_K_M, `-c 32768`, as qA.
* **gI**: granite-4.2-30b-Q5_K_M with `--spec-type ngram-mod` only (no DFlash2 draft): R and v5 must match.

| qA (tokens) | R t/s | v5, first run | v5, re-run | v2 |
|---|---|---|---|---|
| Japanese essay (302) | 28.4 | **+13.3 %** | **+15.0 %** | +10.3 % |
| German story (472) | 33.6 | **+16.9 %** | **+15.9 %** | +13.1 % |
| Rust code (1024) | 102.8 | -1.5 % | 0.0 % | -0.4 % |
| HTML page (1024) | 94.5 | -1.2 % | -1.1 % | -0.3 % |
| 40 capitals, a list (218) | 59.8 | +4.8 % | +5.8 % | -0.1 % |
| role play, turns 1 / 2 / 3 (307 / 302 / 400) | 40.7 / 45.5 / 49.6 | +12.6 / +8.6 / +8.4 % | +12.9 / +8.7 / +9.8 % | +7.5 / -1.2 / +1.8 % |
| geometric mean | | **+7.6 %** | **+8.2 %** | +3.7 % |

The first run's second v5 start is left out: all but one request in it, including those that drafted exactly what the
first v5 start drafted, ran 13-27 % slower than R, so something else held the GPU (it is a desktop machine).  The re-run (R, then v5) is
the replacement.

| qB (tokens) | R t/s | v5 (two starts) | v2 |
|---|---|---|---|
| Japanese essay, T 0 (302) / T 0.7 (386) | 32.3 / 31.5 | +5.7 / +6.1 %, **+13.4 / +14.9 %** | +0.4 %, +4.6 % |
| German story, T 0 (472) / T 0.7 (454) | 38.2 / 39.8 | +5.4 / +5.6 %, -1.1 / -0.5 % | +2.2 %, -0.9 % |
| Chinese story, T 0 (546) / T 0.7 (493) | 31.6 / 31.2 | +5.5 / +6.2 %, +7.4 / +8.2 % | +2.5 %, +4.5 % |
| geometric mean | | **+6.3 %** | +2.2 % |

| mA (tokens) | R t/s | v5 (two starts) | v2 |
|---|---|---|---|
| Japanese essay (512) | 48.5 | +13.1 / +11.6 % | +4.1 % |
| German story (512) | 58.2 | +5.7 / +6.1 % | +0.3 % |
| Rust code (1024) | 58.6 | +10.6 / +10.6 % | +5.6 % |
| Chinese story (747) | 35.9 | **+20.8 / +21.1 %** | +16.3 % |
| zh conversation, turns 1 / 3 (400 / 400) | 52.8 / 55.9 | +5.8 / +5.7 %, +8.8 / +8.8 % | +0.2 %, +0.5 % |
| geometric mean | | **+10.6 %** | +4.4 % |

(The conversation's second turn was a 4-token reply and is left out.)  **gI**: -0.2 to +0.6 % on six requests, the same
text: the switch does nothing without a DFlash2 draft.

### Text

* **Greedy**: every request in every v5 and v2 start gave R's text, with one exception: config B's greedy Chinese story
  on Linux, where v5's text differs from R's (the same text in all four v5 starts; it ran at width 4, and per §63 a verify
  width is a numeric axis).
* **Temperature 0.7**: R reproduced its text in every start.  On Windows each v5 start also reproduced the other's.  On
  Linux v5 did not: seed 1 gave three different texts in four starts, seed 2 two.
* **Why**: v5's decisions use measured times, so a start whose step times differ can move at another draft.  Where the
  width changes the text (sampling, or a near tie under greedy), a v5 run is therefore not reproducible start to start,
  and load from outside the server (as in the left-out Windows start, whose width changes differed from the other start's
  while the text stayed the same) feeds into the width.  v3 and v4 did not have this property; it is the price of item 4.

### Not covered

* The review's own payloads (prose, code-python, recall, reasoning at `-n 192`): v5 was not run on them here.
* `-ncmoe` with the expert-cache arena, `-sm tensor`, and `--parallel 3` or more where the band bound binds: we have one GPU
  per machine and no DFlash2 draft for a MoE target, so the 0.15 default and the band bound ran inert.
* GPUs other than gfx1201.

## Earlier measurements

v3 on r164 (single slot, against fixed 7): a 30K zh continuation +18.7 / +18.4 %, prose -1.8 / -2.7 %, code-python and
recall 0..-0.8 %; the review measured v3 on r171 at -5.9 % on prose and -12 to -15 % on its arena arm.

v2's alignment, `--spec-draft-n-max 5`, two sequences at once on Linux (slot 0 a zh continuation, slot 1 an en
continuation, 384 tokens each), ms per step while both slots verify, means of two starts:

| ms per step, both slots verifying | release | adaptive, not aligned | adaptive, aligned |
|---|---|---|---|
| 8K each | 65.3 | 69.1 (+5.8 %) | 64.2 |
| 32K each | 92.0 | 92.3 | 92.1 |
| 50K each | 112.3 | **108.5 (-3.4 %)** | 112.0 |

The alignment code has not changed since; it stays unconditional, as in `align_draft_caps()`.

*Written by Claude, an AI model by Anthropic, from measurements on the contributor's machines.*
