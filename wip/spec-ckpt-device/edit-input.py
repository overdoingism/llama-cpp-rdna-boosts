#!/usr/bin/env python3
# Report for mmvprof.sh: python3 mmvprof_report.py OUT_DIR
# Decode segments of the kernel trace (split at idle gaps > 3 s: prefill 8K, decode 8K, prefill 110K, decode 110K).  Per target
# step: kernel time by class; the matmul-vector kernels grouped by (kernel, weight type, width, block count), each group mapped
# to the weight matrices of the target / draft GGUF with that type and a row count that fits the block count, and the
# bandwidth that implies.  (Claude, 2026-10-18)
import collections, csv, glob, json, re, struct, sys

OUT = sys.argv[1]
M = "models"
MODELS = {"target": f"{M}/Qwen3.8-27B-UD-Q4_K_XL.gguf", "draft": f"{M}/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"}
PEAK_BW = 620e9   # measured read bandwidth ceiling (mvbw)
# ggml type -> (bytes per block, values per block)
BPB = {0: (4, 1), 1: (2, 1), 30: (2, 1), 2: (18, 32), 6: (22, 32), 8: (34, 32), 12: (144, 256), 13: (176, 256), 14: (210, 256),
       11: (110, 256), 20: (18, 32), 21: (110, 256), 23: (136, 256), 39: (17, 32)}
TNAME = {0: "F32", 1: "F16", 30: "BF16", 2: "Q4_0", 6: "Q5_0", 8: "Q8_0", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
         20: "IQ4_NL", 21: "IQ3_S", 23: "IQ4_XS", 39: "MXFP4"}


def gguf_tensors(path):
    f = open(path, "rb"); buf = f.read(64 << 20); off = 4
    def rd(fmt):
        nonlocal off; v = struct.unpack_from("<" + fmt, buf, off); off += struct.calcsize(fmt); return v[0]
    def rstr():
        nonlocal off; n = rd("Q"); s = buf[off:off + n].decode("utf-8", "replace"); off += n; return s
    SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    def skip(t):
        nonlocal off
        if t == 8: rstr()
        elif t == 9:
            it = rd("I"); n = rd("Q")
            for _ in range(n): skip(it)
        else: off += SZ[t]
    rd("I"); nt = rd("Q"); nkv = rd("Q")
    for _ in range(nkv): rstr(); skip(rd("I"))
    out = []
    for _ in range(nt):
        name = rstr(); nd = rd("I"); dims = [rd("Q") for _ in range(nd)]; t = rd("I"); rd("Q")
        if nd != 2 or not name.endswith(".weight") or "norm" in name or "embd" in name: continue
        bb, bs = BPB.get(t, (1, 1))
        out.append(dict(name=name, type=t, K=dims[0], rows=dims[1], bytes=dims[0] * dims[1] * bb / bs))
    return out


TENS = {m: gguf_tensors(p) for m, p in MODELS.items()}


def generic(name):
    return re.sub(r"^blk\.\d+\.", "blk.N.", name)


# candidate weight matrices per (type, rows): {(model, generic name): (count, bytes each, K)}
CAND = collections.defaultdict(dict)
for m, ts in TENS.items():
    for t in ts:
        k = (m, generic(t["name"]))
        c = CAND[(t["type"], t["rows"])].get(k)
        CAND[(t["type"], t["rows"])][k] = (1 + (c[0] if c else 0), t["bytes"], t["K"])

MMV = re.compile(r"(mul_mat_vec_q(?:_ksplit)?|mul_mat_vec_q_pair\w*|mul_mat_vec_f\w*|lf_\w+|\w*mmv\w*)<\(ggml_type\)(\d+), (\d+)")


def cls(n):
    if MMV.search(n) or "mul_mat_vec" in n: return "matmul-vector"
    for k, c in (("band", "FA band"), ("flash_attn", "FA other"), ("lf_fa", "F8"), ("gated_delta", "DeltaNet"), ("ssm_", "ssm/gdn fused"),
                 ("rms_norm", "norm"), ("l2_norm", "norm"), ("quantize", "quantize"), ("mul_mat_q", "mmq"), ("Cijk_", "hipBLASLt"),
                 ("k_bin_bcast", "binary"), ("unary", "unary"), ("cpy", "copy"), ("concat", "concat"), ("set_rows", "set_rows"),
                 ("rope", "rope"), ("get_rows", "get_rows"), ("soft_max", "softmax"), ("argsort", "top-k"), ("top_k", "top-k")):
        if k in n: return c
    return "other"


def load():
    f = glob.glob(f"{OUT}/prof/**/*kernel_trace.csv", recursive=True)[0]
    rows = []
    with open(f) as fh:
        for r in csv.DictReader(fh):
            gx = int(r.get("Grid_Size_X") or r.get("Grid_Size") or 0); wx = int(r.get("Workgroup_Size_X") or r.get("Workgroup_Size") or 1)
            gy = int(r.get("Grid_Size_Y") or 1); wy = int(r.get("Workgroup_Size_Y") or 1)
            rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"], gx // max(wx, 1), gy // max(wy, 1), wx * wy))
    rows.sort()
    segs, cur = [], [rows[0]]
    for a, b in zip(rows, rows[1:]):
        if b[0] - a[1] > 3e9: segs.append(cur); cur = []
        cur.append(b)
    segs.append(cur)
    return segs


segs = load()
meta = json.load(open(f"{OUT}/prof.json"))
plain = json.load(open(f"{OUT}/plain.json"))
print(f"{len(segs)} segments: " + ", ".join(f"{len(s)}k" if False else f"{len(s)}" for s in segs))
for depth, seg in zip(sorted(meta, key=int), (segs[-3], segs[-1])):
    t = meta[depth]["timings"]; steps = meta[depth]["steps"]
    pt = plain[depth]["timings"]; psteps = plain[depth]["steps"]
    span = (seg[-1][1] - seg[0][0]) / 1e6
    busy, last = 0, 0
    for s, e, *_ in seg:
        if e > last: busy += e - max(s, last); last = e
    print(f"\n== depth {depth}: {t['predicted_n']} tok, {steps} steps, accept {t.get('draft_n_accepted')}/{t.get('draft_n')} | no profiler "
          f"{pt['predicted_ms'] / psteps:.2f} ms/step ({pt['predicted_n'] / psteps:.2f} tok/step) | profiled wall {span / steps:.2f}, "
          f"GPU busy {busy / 1e6 / steps:.2f} ms/step, kernels/step {len(seg) / steps:.0f}")
    c = collections.Counter(); n = collections.Counter()
    for s, e, k, *_ in seg:
        c[cls(k)] += (e - s) / 1e6; n[cls(k)] += 1
    for k, v in c.most_common():
        if v / steps < 0.05: break
        print(f"   {v / steps:7.2f} ms/step  {n[k] / steps:6.1f}/step  {k}")
    g = collections.defaultdict(lambda: [0.0, 0])
    for s, e, k, bx, by, wg in seg:
        if cls(k) != "matmul-vector": continue
        m = re.search(r"(mul_mat_vec_\w+)<\(ggml_type\)(\d+), (\d+), (\w+), (\w+), (\w+), (\d+)", k)
        key = (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(7)), bx, wg) if m else (re.sub(r"\(.*", "", k)[:60], -1, -1, -1, bx, wg)
        g[key][0] += (e - s) / 1e6; g[key][1] += 1
    print(f"   matmul-vector groups: ms/step, n/step (expected), us each, kernel type width rpb rows -> GB/s  [weights]")
    tot_t, tot_b = 0.0, 0.0
    for key, (v, cnt) in sorted(g.items(), key=lambda kv: -kv[1][0]):
        kern, ty, w, rpb, bx, wg = key
        if bx > 1 and (bx - 1) % 256 == 0: bx -= 1   # rdna-boosts grid stall pad: one extra block on multiples of 256
        # rows per block (mmvq-rdna-params.cuh): template 0 = GGML_MMVQ_RDNA4_WEIGHT_RPB (4) for 2..8 columns on one-wave
        # blocks, 2 for the multi-wave Q8_0 short-K block; 1 at one column
        r = rpb if rpb > 0 else (1 if w == 1 else (2 if wg > 32 else 4))
        rows = bx * r
        cands = CAND.get((ty, rows), {})
        exp = sum(cc for cc, _, _ in cands.values())
        desc = "; ".join(f"{mod[0]}:{gname.replace('blk.N.', '')} {rows}x{K} x{cc}" for (mod, gname), (cc, b, K) in cands.items())
        bw = ""
        if cands:
            b = sum(cc * b_ for cc, b_, _ in cands.values()) / exp
            gbs = b * cnt / (v / 1e3) / 1e9
            bw = f" -> {gbs:.0f} GB/s ({100 * gbs * 1e9 / PEAK_BW:.0f}%)"
            tot_t += v; tot_b += b * cnt
        if v / steps < 0.05: continue
        print(f"   {v / steps:6.2f} {cnt / steps:6.1f} ({exp:3d}) {1000 * v / cnt:7.1f}  {kern[-6:]} {TNAME.get(ty, ty)} w{w} r{r} {rows}{bw}  [{desc}]")
    if tot_t:
        print(f"   matched matmul-vector total: {tot_t / steps:.2f} ms/step, {tot_b / steps / 2**30:.2f} GiB/step, "
              f"{tot_b / (tot_t / 1e3) / 1e9:.0f} GB/s")
