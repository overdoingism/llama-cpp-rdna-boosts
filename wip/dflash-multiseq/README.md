# wip/dflash-multiseq — DFlash with `--parallel` ≥ 2: F1 for several sequences, and the readback wait for ubatches without outputs

One `git am` file with two commits, made on `v16-097cf33d9-r50` (net tree `f102245d…`; with both commits applied, tree `46170619…`). Both commits apply on their own to plain r50 (`git apply --check`). They touch `src/llama-context.cpp`, `src/llama-context.h`, `src/llama-ext.h` and `common/speculative.cpp`; nothing under `ggml/`.

1. **F1 for several sequences.** Until now F1 (target features kept on the device) was single-sequence only, so a DFlash server with `--parallel` ≥ 2 took the host path and lost its prefill speed.
   - A multi-sequence batch is split by `split_equal`, so the ubatch order can differ from the batch order. `output_reorder()` puts only the host rows back.
   - The target now keeps the inverse of `embd_batch_idxs` (`llama_lf_get_dev_rows`; nullptr when the order held, which is always the case for one sequence). The draft maps each batch row to its device row.
   - The host rows are no longer permuted on the F1 path, where nothing reads them.
   - Exception: when the target has host-resident MoE experts (`-ncmoe`/`-cmoe`), several sequences keep the host path. With the movable-boundary slab armed for them, the draft's device reads went wrong (see the PR text; related to #127). One sequence is unchanged.
2. **`GGML_READBACK_SYNC` also waits in a ubatch without outputs when per-token readbacks ran.** These readbacks are the DFlash layer inputs (host path, or F1's device copy) and the unmasked nextn embeddings. They run for every token, also in a prompt chunk that does not end its prompt. The next ubatch's input upload only waits for this graph's compute, not for the copies queued after it.

Both are output-neutral for the target model; the draft gets the same feature values as with one sequence.

*The patch and this note were written by Claude (AI).*
