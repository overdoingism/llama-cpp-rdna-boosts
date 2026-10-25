# wip/fa-band-skip — the decode/verify band skips fully masked KV groups

One `git am` file (one commit), made on `v14-3f03b1772-r131` (tree `5946a27a…`; applied: tree `8e5ee878…`). It touches
only `ggml/src/ggml-cuda/fattn-common.cuh` and `fattn-mma-f16.cuh`.

- With a unified KV cache a sequence's attention covers every cell up to its last one, including the masked cells of
  other slots. The fully masked group skip (issue #48) runs for prefill only, so the band (n_q <= 8) reads all of them.
  Since #210 (r130) keeps idle slots resident, a conversation whose cells come after a long idle one pays for those
  cells on every decode step.
- The band loop already handles the packed group bitmap (`kq_blocks`). The launcher now builds it for the band too
  (`flash_attn_mask_to_KV_blocks`, per stream and query tile), and the band entry passes its tile's row instead of
  `nullptr`.
- Block i still owns iterations i, i+P, i+2P, ... and only leaves out the ones whose logits are all -INF, so the
  accumulation order over the rest is unchanged and the result is bit-identical. `GGML_CUDA_FA_BAND_SKIP=0` turns it
  off. RDNA4 only, as the band is.

Measured on gfx1201 with r131, Qwen3.8-27B UD-Q4_K_XL + DFlash2, 2 slots, unified q8_0 KV:

- with a ~100K context resident on the other slot, an 8.6K-token request decodes at 57.44 -> 47.74 ms per step
  (-16.9 %), with identical token ids;
- `test-backend-ops -o FLASH_ATTN_EXT` (head 256, masked, q8_0 / f16 K/V) passes 292/292.

The relation to #210, the alternatives, and the earlier r115 measurements are in the PR description.

*The patch and this note were written by Claude (AI).*
