"""
GLM-5.2-FP8 -> our int4 container converter (STAGE B).

DISK-SAFE strategy (user request): download ONE shard (~5 GB), convert it to int4,
DELETE it, move on to the next. The disk never fills up: peak = 1 shard + the int4
output growing up to ~372 GB. A space check stops if the margin runs low.

What it does for each tensor:
  - FP8 (e4m3) weights with `*.weight_scale_inv`  -> dequant in 128x128 blocks -> f32
  - BF16 weights (norms/embed/lm_head/...)         -> f32
  then:
  - attn/mlp/shared/expert/embed/lm_head -> QUANTIZED to int4 (or int8) with the SAME math
    as the C engine (np.rint = lrintf, same thresholds, same nibble packing) -> identical tokens
  - norms / router (mlp.gate.weight) / bias / e_score_correction_bias -> kept F32
  - DSA indexer / MTP layer (78) / shared_head / eh_proj / indexer *norm -> SKIPPED

Output: a directory of safetensors readable by the C engine (for each quantized weight:
`name` U8 = packed data, `name.qs` F32 = per-row scales).

USAGE:
  # local test (tiny oracle, no download): converts a directory that already exists
  python3 tools/convert_fp8_to_int4.py --indir glm_tiny --outdir glm_tiny_i4 --ebits 4 --io-bits 4
  # selftest of the fp8 dequant (requires torch)
  python3 tools/convert_fp8_to_int4.py --selftest
  # real: download+convert+delete shard by shard
  python3 tools/convert_fp8_to_int4.py --repo zai-org/GLM-5.2-FP8 --outdir /home/vincenzo/glm52_i4
"""
import os, sys, glob, json, shutil, argparse
import numpy as np

# ---------- quantization: identical to the C code (glm.c) ----------
def quant_int8(w, bits):                       # w: [O,I] f32 -> (qbytes U8 [O*I], scale f32 [O])
    qmax = (1 << (bits - 1)) - 1
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = np.maximum(amax / qmax, 1e-8)
    q = np.clip(np.rint(w / s), -qmax - 1, qmax).astype(np.int8)
    return q.reshape(-1).view(np.uint8).copy(), s[:, 0].astype(np.float32)

def quant_int4(w, bits):                        # -> (qbytes U8 [O*ceil(I/2)], scale f32 [O])
    O, I = w.shape
    qmax = (1 << (bits - 1)) - 1
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = np.maximum(amax / qmax, 1e-8)
    q = np.clip(np.rint(w / s), -8, qmax).astype(np.int32)  # nibbles [-8,7]
    rb = (I + 1) // 2
    out = np.zeros((O, rb), np.uint8)
    v0 = (q[:, 0::2] + 8).astype(np.uint8)
    out[:, :v0.shape[1]] = v0
    if I > 1:
        v1 = (q[:, 1::2] + 8).astype(np.uint8)
        out[:, :v1.shape[1]] |= (v1 << 4)
    return out.reshape(-1), s[:, 0].astype(np.float32)

def quant_int4_grouped(w, bits, gs=128):
    """Group-scaled int4: one scale per group of `gs` elements along the input dim.
    Drastically reduces quantization error vs per-row scaling — matches the FP8
    source's 128x128 block-scale granularity. Output layout:
      qbytes: same packed nibble format as quant_int4
      scales: f32 [O * ngroups] where ngroups = ceil(I/gs), laid out as
              s[o * ngroups + g] = scale for row o, group g.
    The engine detects this format (fmt=4) by checking the .qs array size."""
    O, I = w.shape
    qmax = (1 << (bits - 1)) - 1
    ngroups = (I + gs - 1) // gs
    # pad I to a multiple of gs for clean reshape, then trim
    Ipad = ngroups * gs
    wpad = np.zeros((O, Ipad), np.float32)
    wpad[:, :I] = w
    wr = wpad.reshape(O, ngroups, gs)                     # [O, ngroups, gs]
    amax = np.abs(wr).max(axis=2, keepdims=True)          # [O, ngroups, 1]
    s = np.maximum(amax / qmax, 1e-8)                     # [O, ngroups, 1]
    q = np.clip(np.rint(wr / s), -8, qmax).astype(np.int32)  # [O, ngroups, gs]
    q = q.reshape(O, Ipad)[:, :I]                         # trim padding -> [O, I]
    # pack nibbles (identical to quant_int4)
    rb = (I + 1) // 2
    out = np.zeros((O, rb), np.uint8)
    v0 = (q[:, 0::2] + 8).astype(np.uint8)
    out[:, :v0.shape[1]] = v0
    if I > 1:
        v1 = (q[:, 1::2] + 8).astype(np.uint8)
        out[:, :v1.shape[1]] |= (v1 << 4)
    # scales: flatten [O, ngroups] -> [O * ngroups]
    s_flat = s[:, :, 0].astype(np.float32).reshape(-1)
    return out.reshape(-1), s_flat

def quant_int2(w, bits):                        # -> (qbytes U8 [O*ceil(I/4)], scale f32 [O]); 4/byte
    O, I = w.shape
    qmax = (1 << (bits - 1)) - 1                 # bits=2 -> qmax=1, values [-2,1]
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = np.maximum(amax / qmax, 1e-8)
    q = np.clip(np.rint(w / s), -2, qmax).astype(np.int32)
    rb = (I + 3) // 4
    out = np.zeros((O, rb), np.uint8)
    for k in range(4):                           # pack 4 values per byte (identical to pack_int2 in C)
        vk = q[:, k::4]
        out[:, :vk.shape[1]] |= ((vk + 2).astype(np.uint8) << (k * 2))
    return out.reshape(-1), s[:, 0].astype(np.float32)

# ---------- NVFP4 (modelopt) : e2m1 LUT ----------
# FP4 e2m1 = 1 sign + 2 exp + 1 mantissa. 16 codes, magnitudes {0,.5,1,1.5,2,3,4,6}.
# Bit 3 = sign. Packed order (compressed_tensors/vLLM): LOW nibble = even element,
# HIGH nibble = odd element. LUT verified 1:1 against ml_dtypes.float4_e2m1fn.
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

# ---------- tensor classification ----------
def layer_idx(name):
    p = name.split(".")
    if len(p) > 2 and p[0] == "model" and p[1] == "layers":
        try: return int(p[2])
        except ValueError: return -1
    return -1

def classify(name, n_layers, keep_mtp=False, keep_idx=False):
    if name.endswith("_scale_inv"): return "consumed"   # FP8 base: handled with its weight
    # NVFP4 (modelopt): scale sidecars are consumed together with their U8 .weight.
    if name.endswith((".weight_scale", ".weight_scale_2", ".input_scale")): return "consumed"
    li = layer_idx(name)
    if keep_idx:
        # --indexer mode: ONLY the DSA lightning-indexer weights of the main layers
        if li < 0 or li >= n_layers or "indexer" not in name: return "skip"
        if name.endswith("norm.weight"): return "f32"
        return "q"                                       # int8 recommended (--ebits 8): scoring weights
    if keep_mtp:
        if li != n_layers: return "skip"                 # only the MTP layer
        if "indexer" in name: return "skip"              # the DSA indexer stays a no-op
    else:
        if li >= n_layers: return "skip"                 # MTP layer (78)
        if any(k in name for k in ["indexer", "indexers_proj", "eh_proj",
                                    "enorm", "hnorm", "shared_head"]): return "skip"
    if name.endswith("e_score_correction_bias"): return "f32"
    if name.endswith("mlp.gate.weight"): return "f32"    # router (NOT gate_proj)
    if name.endswith("norm.weight") or name == "model.norm.weight": return "f32"
    if name in ("model.embed_tokens.weight", "lm_head.weight"): return "io"
    if ".mlp.experts." in name and name.endswith(".weight"): return "x"  # ROUTED expert (streaming)
    # Split resident weights by type for mixed-precision control:
    #   "sh" = shared expert (fires on every token, highest sensitivity)
    #   "o"  = o_proj attention (reconstructs output, biggest attn tensor)
    #   "kvb" = kv_b_proj (reconstructs KV cache on every decode step)
    #   "attn" = other attention projections (q_a, q_b, kv_a)
    #   "dmlp" = dense MLP (first 3 layers)
    if "shared_experts" in name: return "sh"
    if name.endswith("o_proj.weight"): return "o"
    if name.endswith("kv_b_proj.weight"): return "kvb"
    if any(name.endswith(k) for k in ("q_a_proj.weight", "q_b_proj.weight",
                                       "kv_a_proj_with_mqa.weight")): return "attn"
    if any(name.endswith(k) for k in ("mlp.gate_proj.weight", "mlp.up_proj.weight",
                                       "mlp.down_proj.weight")): return "dmlp"
    if name.endswith(".weight"): return "q"              # fallback: other resident weights
    return "f32"

# ---------- NVFP4 (modelopt) dequant of ONE expert tensor -> f32 [O,I] ----------
def dequant_nvfp4(f, name):
    """NVIDIA modelopt NVFP4 (quant_algo=NVFP4, quant_method=modelopt).
      - `name`               U8   [O, I/2]  : two e2m1 nibbles per byte along the
                                              contraction dim (input); even=low nibble, odd=high.
      - `name.weight_scale`  F8_E4M3 [O, I/16] : per-BLOCK scale of 16 elements (group_size=16),
                                              along the input dim. Decodes f8e4m3 -> f32.
      - `name.weight_scale_2` F32 []        : GLOBAL per-tensor scale, ~amax/(6*448) (small).
    Dequant (modelopt convention = MULTIPLY, NOT divide):
        W[o,i] = e2m1_lut[nibble] * f8_block_scale[o, i//16] * weight_scale_2
    FOOTGUN: llm-compressor/compressed-tensors stores the RECIPROCAL (large global) and DIVIDES;
    modelopt stores the small value and MULTIPLIES. This checkpoint is modelopt -> multiply."""
    import torch
    GS = 16                                                       # NVFP4: block scale every 16 elements
    packed = f.get_tensor(name)                                    # uint8 [O, I/2]
    bscale = f.get_tensor(name + "_scale").to(torch.float32)        # [O, ceil(I/16)] from f8e4m3
    gscale = f.get_tensor(name + "_scale_2").to(torch.float32)      # per-tensor scalar
    O, Ih = packed.shape; I = Ih * 2
    # Convention: modelopt stores the SMALL global and MULTIPLIES. If it is >=1 it is
    # almost certainly the compressed-tensors reciprocal (which DIVIDES) -> we stop
    # instead of silently corrupting every tensor.
    assert float(gscale) < 1.0, (
        f"{name}: weight_scale_2={float(gscale):.4g} >= 1 looks like the reciprocal "
        "(compressed-tensors, which DIVIDES); this path assumes modelopt (MULTIPLIES)")
    # The layout must be modelopt's flat per-block scale: one column every 16 input
    # elements (no cutlass/TensorRT swizzle). We verify, we don't infer:
    # inferring gs = I // ncol silently misaligns on padded/swizzled layouts.
    nb = (I + GS - 1) // GS
    assert bscale.shape[1] == nb, (
        f"{name}: weight_scale has {bscale.shape[1]} columns, expected {nb} = ceil({I}/{GS}); "
        "unexpected scale layout (swizzled/padded?), refusing so as not to corrupt")
    lut = torch.tensor(_E2M1, dtype=torch.float32)
    nib = torch.empty((O, I), dtype=torch.long)
    nib[:, 0::2] = (packed & 0x0F).to(torch.long)                  # even element = low nibble
    nib[:, 1::2] = ((packed >> 4) & 0x0F).to(torch.long)           # odd element = high nibble
    w4 = lut[nib]                                                  # [O, I] e2m1 values
    sc = bscale.repeat_interleave(GS, dim=1)[:, :I]               # partial tail block: slice to I
    return (w4 * sc * gscale).numpy()

# ---------- dequant of one tensor (nvfp4 / fp8+block-scale / bf16 / f32) ----------
def dequant(f, name, keys):
    import torch
    sl = f.get_slice(name); dt = sl.get_dtype()
    # NVFP4 (modelopt): U8 expert weights with a `.weight_scale` sidecar. In this checkpoint
    # the ONLY U8 tensors are the NVFP4 experts, but we still require the sidecar (keys is
    # mandatory: without it, any U8 would be decoded as NVFP4).
    if dt in ("U8", "uint8") and (name + "_scale") in keys:
        return dequant_nvfp4(f, name)
    if dt in ("F8_E4M3", "float8_e4m3fn"):
        w = f.get_tensor(name).to(torch.float32)
        sc = f.get_tensor(name + "_scale_inv").to(torch.float32)   # [ceil(O/128),ceil(I/128)]
        O, I = w.shape
        sc = sc.repeat_interleave(128, 0).repeat_interleave(128, 1)[:O, :I]
        return (w * sc).numpy()
    return f.get_tensor(name).to(torch.float32).numpy()

def convert_shard(path, out_dict, n_layers, ebits, io_bits, xbits,
                  keep_mtp=False, keep_idx=False, group_size=0, bits_map=None):
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        keys = set(f.keys())
        for name in f.keys():
            kind = classify(name, n_layers, keep_mtp, keep_idx)
            if kind in ("skip", "consumed"): continue
            w = dequant(f, name, keys)
            if kind == "f32":
                out_dict[name] = w.astype(np.float32)
            else:
                # Resolve bits for this tensor type: use bits_map override if provided,
                # otherwise fall back to the classic ebits/xbits/io_bits scheme.
                if bits_map and kind in bits_map:
                    bits = bits_map[kind]
                else:
                    bits = io_bits if kind == "io" else xbits if kind == "x" else ebits
                # Any unknown kind that fell through classify as "q"
                if bits_map and kind not in bits_map and kind not in ("io", "x", "sh", "o", "kvb", "attn", "dmlp"):
                    bits = ebits
                if w.ndim != 2:        # e.g. a 1D bias not expected as 'q' -> keep it f32
                    out_dict[name] = w.astype(np.float32); continue
                if group_size > 0 and bits <= 4:
                    q, s = quant_int4_grouped(w, bits, group_size)
                else:
                    q, s = (quant_int2(w, bits) if bits <= 2 else
                            quant_int4(w, bits) if bits <= 4 else quant_int8(w, bits))
                out_dict[name] = q
                out_dict[name + ".qs"] = s

def free_gb(p): return shutil.disk_usage(p).free / 1e9

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    ap.add_argument("--indir", default=None)
    ap.add_argument("--outdir", required=False)
    ap.add_argument("--ebits", type=int, default=None)   # resident bits (default 4; 8 for --mtp/--indexer)
    ap.add_argument("--io-bits", type=int, default=8)    # bits for embed/lm_head
    ap.add_argument("--xbits", type=int, default=None)   # bits for ROUTED experts (streaming); default=ebits
    # Mixed-precision: per-tensor-type bit overrides. Default = ebits (all same).
    # Set these higher to protect sensitive tensors from quantization error.
    ap.add_argument("--shared-bits", type=int, default=None,
        help="bits for shared expert (fires on every token, highest sensitivity). Default=ebits")
    ap.add_argument("--o-bits", type=int, default=None,
        help="bits for o_proj attention (reconstructs output, biggest attn tensor). Default=ebits")
    ap.add_argument("--kvb-bits", type=int, default=None,
        help="bits for kv_b_proj (reconstructs KV cache on every decode). Default=ebits")
    ap.add_argument("--attn-bits", type=int, default=None,
        help="bits for other attention projections (q_a, q_b, kv_a). Default=ebits")
    ap.add_argument("--dmlp-bits", type=int, default=None,
        help="bits for dense MLP (first 3 layers). Default=ebits")
    ap.add_argument("--group-size", type=int, default=0,  # 0 = per-row (backward compat); 128 = group-scaled
        help="group size for int4 scales: 0=per-row (default), 128=one scale per 128 elements (much better quality)")
    ap.add_argument("--n-layers", type=int, default=78)
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-nvfp4", action="store_true",
        help="unit-test the NVFP4 dequant (e2m1 LUT + round-trip), no download / no network")
    ap.add_argument("--mtp", action="store_true",
        help="download and convert ONLY the MTP head (model.layers.<n_layers>.*) -> out-mtp-*.safetensors")
    ap.add_argument("--indexer", action="store_true",
        help="extract ONLY the DSA lightning-indexer weights -> out-idx-*.safetensors. WARNING: "
             "indexer tensors are spread across nearly every shard, so this re-downloads the whole "
             "repository (~756 GB of traffic) to retain only a few GB. Resumable per shard. "
             "Recommended: --ebits 8.")
    a = ap.parse_args()
    if a.ebits is None:
        # MTP head at int4 = acceptance ~0-4% (measured, issue #8): the draft is always wrong
        # and speculation never kicks in. At int8: 39-59%, 2.2-2.8 tokens/forward.
        a.ebits = 8 if (a.mtp or a.indexer) else 4
    if a.xbits is None: a.xbits = a.ebits

    # Build per-type bits map. If a type-specific arg is set, use it; otherwise the
    # converter falls back to ebits for that type.
    bits_map = {}
    if a.shared_bits is not None: bits_map["sh"] = a.shared_bits
    if a.o_bits is not None:      bits_map["o"] = a.o_bits
    if a.kvb_bits is not None:    bits_map["kvb"] = a.kvb_bits
    if a.attn_bits is not None:   bits_map["attn"] = a.attn_bits
    if a.dmlp_bits is not None:   bits_map["dmlp"] = a.dmlp_bits
    if bits_map:
        print(f"[MIXED] precision map: " + ", ".join(f"{k}={v}bit" for k,v in sorted(bits_map.items())))

    if a.selftest_nvfp4:
        import torch
        # 1) e2m1 LUT: the 16 codes must decode exactly to the expected values.
        lut = torch.tensor(_E2M1, dtype=torch.float32)
        expect = [0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,-0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0]
        assert lut.tolist() == expect, "wrong e2m1 LUT"
        print("[nvfp4] e2m1 LUT: 16/16 codes OK")
        # 2) round-trip: build a tensor using ONLY representable values (known per-block +
        #    global scale), pack it like modelopt, then dequant must come back EXACT.
        import numpy as np, io
        from safetensors.torch import save as st_save
        from safetensors import safe_open
        rng = np.random.default_rng(0); O, I, GS = 8, 64, 16
        codes = rng.integers(0, 16, size=(O, I)).astype(np.uint8)   # random e2m1 nibbles
        w4 = np.array(_E2M1, np.float32)[codes]                      # [O,I]
        # per-block scales (representable in f8e4m3) + small global (modelopt style)
        blk = rng.choice([0.5,1.0,2.0,4.0,8.0], size=(O, I//GS)).astype(np.float32)
        gscale = np.float32(3.9e-5)
        W = w4 * np.repeat(blk, GS, axis=1) * gscale                 # exact reference
        # pack: even->low nibble, odd->high
        packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
        import ml_dtypes  # test only: f8e4m3 encode of the block scales
        tens = {name: torch.from_numpy(arr) for name, arr in {
            "w.weight": packed,
            "w.weight_scale": blk.astype(ml_dtypes.float8_e4m3fn).view(np.uint8),  # placeholder
        }.items()}
        # torch has no constructor from f8 bytes: go through a hand-written safetensors file.
        # simpler: call dequant_nvfp4 directly on a fake in-memory 'f'.
        class _F:
            def __init__(s, d): s.d = d
            def get_tensor(s, n): return s.d[n]
            def get_slice(s, n): return None
        blk_f8 = blk.astype(ml_dtypes.float8_e4m3fn)                 # quantize the scales to f8
        f = _F({"w.weight": torch.from_numpy(packed),
                "w.weight_scale": torch.from_numpy(blk_f8.view(np.uint8)).view(torch.float8_e4m3fn),
                "w.weight_scale_2": torch.tensor(gscale)})
        got = dequant_nvfp4(f, "w.weight")
        # reference with scales already quantized to f8 (for an exact comparison)
        Wq = w4 * np.repeat(blk_f8.astype(np.float32), GS, axis=1) * gscale
        maxerr = float(np.abs(got - Wq).max())
        print(f"[nvfp4] round-trip encode->dequant: max abs err = {maxerr:.3e} "
              f"({'OK' if maxerr < 1e-9 else 'FAIL'})")
        assert maxerr < 1e-9
        # 3) colibri int4 requant on the dequantized values -> small error expected
        q, s = quant_int4(got.astype(np.float32), 4)
        rb = (I + 1)//2; qb = q.reshape(O, rb)
        lo = (qb & 0x0F).astype(np.int32) - 8; hi = ((qb >> 4) & 0x0F).astype(np.int32) - 8
        deq = np.empty((O, I), np.float32); deq[:, 0::2] = lo; deq[:, 1::2] = hi[:, :I-I//2]
        deq = deq * s[:, None]
        rel = np.abs(deq - got).mean() / (np.abs(got).mean() + 1e-12)
        # Informational, NOT an equality test: per-row int4 requant of data spanning 16x
        # because of the block-scale costs ~0.17 relative error on its own. The loose
        # threshold only catches gross corruption, it is not a precision bound.
        print(f"[nvfp4] dequant->colibri int4->dequant: mean rel error = {rel:.4f} "
              f"(expected ~0.17; {'OK' if rel < 0.30 else 'ANOMALOUS'})")
        assert rel < 0.30, f"requant rel err {rel:.3f} too high: dequant probably corrupted"
        print("[nvfp4] SELFTEST OK")
        return

    if a.selftest:
        import torch
        w = (torch.randn(256, 256) * 0.3)
        O, I = w.shape; bs = 128
        sc = torch.zeros(O // bs, I // bs)
        for bi in range(O // bs):
            for bj in range(I // bs):
                blk = w[bi*bs:(bi+1)*bs, bj*bs:(bj+1)*bs]
                sc[bi, bj] = blk.abs().max() / 448.0
        q = (w / sc.repeat_interleave(bs,0).repeat_interleave(bs,1)).to(torch.float8_e4m3fn)
        deq = (q.to(torch.float32) * sc.repeat_interleave(bs,0).repeat_interleave(bs,1))
        rel = (deq - w).abs().mean() / w.abs().mean()
        print(f"[selftest fp8 block-dequant] mean relative error = {rel:.4f}  "
              f"({'OK' if rel < 0.05 else 'HIGH'})")
        return

    os.makedirs(a.outdir, exist_ok=True)
    if a.indir:    # local conversion (test)
        shards = sorted(glob.glob(os.path.join(a.indir, "*.safetensors")))
        from safetensors.numpy import save_file
        for i, sp in enumerate(shards):
            out = {}; convert_shard(sp, out, a.n_layers, a.ebits, a.io_bits, a.xbits, group_size=a.group_size, bits_map=bits_map)
            save_file(out, os.path.join(a.outdir, f"out-{i:05d}.safetensors"))
        # copy config + tokenizer
        for fn in ["config.json"]:
            src = os.path.join(a.indir, fn)
            if os.path.exists(src): shutil.copy(src, a.outdir)
        print(f"converted {len(shards)} shards -> {a.outdir}")
        return

    # real: download shard by shard, convert, delete
    #
    # NETWORK ROBUSTNESS: short read timeouts so a hung download FAILS instead of
    # sitting there forever. 8s, not 30: a "timeout" means ZERO bytes received in that
    # window; on a live transfer chunks arrive constantly, so 8s is safe and a stall
    # costs 8s instead of 30.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "8")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "15")
    # timestamped logs: hf_hub's "Trying to resume" messages become datable.
    import logging
    logging.basicConfig(format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    # hf_xet hangs when the network restarts (zombie connections with no timeout):
    # force the classic HTTP path, which curl proved works (measured 2026-07-02).
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # =0 to re-enable xet
    from huggingface_hub import HfApi, hf_hub_download

    # anti-duplicate lock: TWO converters on the same outdir corrupt each other.
    # fcntl is Unix-only; on Windows use msvcrt or skip locking.
    lock = open(os.path.join(a.outdir, ".convert.lock"), "w")
    try:
        import fcntl
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("ERROR: another converter is already using this output directory. Exiting."); return
    except ImportError:
        try:
            import msvcrt
            try: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                print("ERROR: another converter is already using this output directory. Exiting."); return
        except ImportError:
            pass  # no locking available — single-user converter, acceptable

    # known file sizes, filled in after repo_info: the multi-stream downloader uses them
    # to compute segment boundaries and to know when a file is complete.
    SIZES = {}

    def download_retry(repo, fn, dest, tries=999):
        """Multi-stream downloader with Range resume. Opens N concurrent segments
        (default 2, COLI_DL_STREAMS to change) and saves per-segment state in a .seg
        sidecar -> NO byte is lost however the connection dies. A single HF stream is
        paced at ~2 MB/s (measured); 2 streams roughly double throughput without
        saturating a home line. Small files, COLI_DL_STREAMS=1 or an old legacy .part
        -> single-stream path (_download_single)."""
        import time as _t, threading, urllib.request, urllib.error
        url = f"https://huggingface.co/{repo}/resolve/main/{fn}"
        out = os.path.join(dest, fn); part = out + ".part"; side = part + ".seg"
        os.makedirs(dest, exist_ok=True)
        expected = SIZES.get(fn)
        if os.path.exists(out) and (expected is None or os.path.getsize(out) == expected):
            return out
        NS = max(1, min(8, int(os.environ.get("COLI_DL_STREAMS", "2"))))
        # a .part without a sidecar was written by an older single-stream version.
        legacy = os.path.exists(part) and not os.path.exists(side)
        if expected is None or expected < (256 << 20) or NS == 1 or legacy:
            return _download_single(url, fn, out, part, expected)
        # ---- multi-stream ----
        segs = [(expected * t // NS, expected * (t + 1) // NS) for t in range(NS)]
        done = [0] * NS
        # resume per-segment progress if the sidecar matches (same N, same size).
        if os.path.exists(side):
            try:
                st = json.loads(open(side).read())
                if st.get("n") == NS and st.get("size") == expected: done = st["done"]
            except Exception: pass
        if not os.path.exists(part):
            with open(part, "wb") as f: f.truncate(expected)   # sparse file
        fd = os.open(part, os.O_WRONLY)
        t0 = _t.time(); nres = [0]; log_lock = threading.Lock(); stopfail = []
        def worker(t):
            s0, s1 = segs[t]
            while done[t] < s1 - s0 and not stopfail:
                pos = s0 + done[t]
                req = urllib.request.Request(url, headers={"User-Agent": "colibri-convert",
                                                           "Range": f"bytes={pos}-{s1-1}"})
                try:
                    with urllib.request.urlopen(req, timeout=8) as r:
                        if r.status != 206:               # Range ignored: multi-stream impossible
                            stopfail.append(t); return
                        while done[t] < s1 - s0:
                            chunk = r.read(1 << 20)
                            if not chunk: break
                            rem = (s1 - s0) - done[t]     # never past the segment
                            if len(chunk) > rem: chunk = chunk[:rem]
                            os.pwrite(fd, chunk, s0 + done[t])
                            done[t] += len(chunk)
                except KeyboardInterrupt: raise
                except Exception as ex:
                    with log_lock:
                        nres[0] += 1
                        print(f"    [dl] s{t}: {type(ex).__name__} at {(s0+done[t])/1e9:.2f} GB: "
                              f"resuming (#{nres[0]})", flush=True)
                    _t.sleep(min(15, 1 + nres[0] // NS))
        th = [threading.Thread(target=worker, args=(t,), daemon=True) for t in range(NS)]
        for x in th: x.start()
        print(f"    [dl {_t.strftime('%H:%M:%S')}] connected: {NS} streams, "
              f"{sum(done)/1e9:.2f} of {expected/1e9:.2f} GB", flush=True)
        mark = sum(done); tmark = t0
        while any(x.is_alive() for x in th):
            _t.sleep(5)
            have = sum(done)
            tmpside = side + ".tmp"                       # atomic checkpoint
            open(tmpside, "w").write(json.dumps({"n": NS, "size": expected, "done": list(done)}))
            os.replace(tmpside, side)
            now = _t.time()
            if now - tmark >= 30:
                print(f"    [dl {_t.strftime('%H:%M:%S')}] {have/1e9:5.2f} GB "
                      f"({(have-mark)/max(now-tmark,1e-9)/1e6:5.1f} MB/s, {NS} stream)", flush=True)
                mark = have; tmark = now
        os.close(fd)
        if stopfail:                                      # server won't honor Range: fall back
            for f2 in (part, side):
                if os.path.exists(f2): os.remove(f2)
            return _download_single(url, fn, out, part, expected)
        assert sum(done) == expected
        if os.path.exists(side): os.remove(side)
        os.replace(part, out)
        dt = max(_t.time() - t0, 1e-9)
        print(f"    [dl] {fn}: {expected/1e9:.2f} GB in {dt/60:.1f} min "
              f"({expected/dt/1e6:.1f} MB/s avg, {NS} streams, {nres[0]} resumes)", flush=True)
        return out

    def _download_single(url, fn, out, part, expected):
        """Single-stream path with Range resume (small files / legacy .part /
        COLI_DL_STREAMS=1). A clean short EOF counts as a resume; if NO new byte
        arrives, back off instead of spinning."""
        import time as _t, urllib.request, urllib.error
        t0 = _t.time(); nres = 0; mark = 0; tmark = t0
        while True:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            if expected is not None and have >= expected: break
            have0 = have
            req = urllib.request.Request(url, headers={"User-Agent": "colibri-convert"})
            if have: req.add_header("Range", f"bytes={have}-")
            try:
                with urllib.request.urlopen(req, timeout=8) as r:
                    if have and r.status == 200:          # server ignored Range: restart clean
                        have = 0
                    if expected is None:
                        cl = r.headers.get("Content-Length")
                        if cl: expected = have + int(cl)
                    if have == 0 or nres:                 # immediate sign of life
                        print(f"    [dl {_t.strftime('%H:%M:%S')}] connected"
                              f"{f' @ {have/1e9:.2f} GB' if have else ''}"
                              f"{f' of {expected/1e9:.2f} GB' if expected else ''}", flush=True)
                    with open(part, "ab" if have else "wb") as f:
                        if not have: f.truncate(0)
                        while True:
                            chunk = r.read(1 << 20)
                            if not chunk: break
                            f.write(chunk); have += len(chunk)
                            if have - mark >= 512 * 1024 * 1024 or _t.time() - tmark >= 30:
                                now = _t.time()
                                print(f"    [dl {_t.strftime('%H:%M:%S')}] {have/1e9:5.2f} GB "
                                      f"({(have-mark)/max(now-tmark,1e-9)/1e6:5.1f} MB/s)", flush=True)
                                mark = have; tmark = now
                if expected is None: break                # unknown length: single pass
                if have < expected:                       # clean short EOF: counts as a resume
                    nres += 1
                    if have == have0: _t.sleep(min(15, 1 + nres))   # zero progress -> back off
            except KeyboardInterrupt: raise
            except urllib.error.HTTPError as ex:
                if ex.code == 416: break                  # already complete
                nres += 1
                print(f"    [dl] HTTP {ex.code} at {have/1e9:.2f} GB: resuming (#{nres})", flush=True)
                _t.sleep(min(15, 1 + nres))
            except Exception as ex:
                nres += 1
                print(f"    [dl] {type(ex).__name__} at {have/1e9:.2f} GB: resuming (#{nres})", flush=True)
                _t.sleep(min(15, 1 + nres))
        os.replace(part, out)
        dt = max(_t.time() - t0, 1e-9); sz = os.path.getsize(out)
        print(f"    [dl] {fn}: {sz/1e9:.2f} GB in {dt/60:.1f} min "
              f"({sz/dt/1e6:.1f} MB/s avg, {nres} resumes)", flush=True)
        return out

    from safetensors.numpy import save_file
    import time as _t
    info = None
    for att in range(10):
        try:
            info = HfApi().repo_info(a.repo, files_metadata=True)
            # sizes known from the store: enable segmented multi-stream download.
            SIZES.update({s.rfilename: s.size for s in info.siblings if s.size})
            break
        except KeyboardInterrupt: raise
        except Exception as ex:
            w = min(60, 5*(att+1)); print(f"repo_info failed ({type(ex).__name__}); retrying in {w}s", flush=True); _t.sleep(w)
    if info is None:
        print("ERROR: could not reach the repository after 10 retries. Check your network and repo name.", flush=True)
        return
    shards = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(".safetensors"))
    if not shards:
        print("ERROR: no .safetensors shards found in this repository.", flush=True)
        return
    for fn in ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"]:
        try: shutil.copy(hf_hub_download(a.repo, fn, local_dir=a.outdir+"/_meta"), a.outdir)
        except Exception: pass
    tmp = os.path.join(a.outdir, "_inflight"); os.makedirs(tmp, exist_ok=True)
    if a.mtp:
        import urllib.request
        idx = json.loads(urllib.request.urlopen(
            f"https://huggingface.co/{a.repo}/resolve/main/model.safetensors.index.json", timeout=30).read())["weight_map"]
        pref = f"model.layers.{a.n_layers}."
        mtp_shards = sorted(set(v for k, v in idx.items() if k.startswith(pref)))
        print(f"[MTP] head at layer {a.n_layers}: {len(mtp_shards)} shards to process: {mtp_shards}")
        for i, sh in enumerate(mtp_shards):
            outp = os.path.join(a.outdir, f"out-mtp-{i:05d}.safetensors")
            if os.path.exists(outp): print(f"[MTP] {outp} already done"); continue
            print(f"[MTP {i+1}/{len(mtp_shards)}] downloading {sh}...", flush=True)
            p = download_retry(a.repo, sh, tmp)
            out = {}; convert_shard(p, out, a.n_layers, a.ebits, a.io_bits, a.xbits, keep_mtp=True, group_size=a.group_size, bits_map=bits_map)
            save_file(out, outp)
            os.remove(p)
            for blob in glob.glob(os.path.join(tmp, "**", "*"), recursive=True):
                if os.path.isfile(blob): os.remove(blob)
            print(f"    -> {os.path.basename(outp)} ({os.path.getsize(outp)/1e9:.2f} GB, {len(out)} tensors)", flush=True)
        shutil.rmtree(tmp, ignore_errors=True); print("[MTP] DONE."); return
    if a.indexer:
        import urllib.request
        idx = json.loads(urllib.request.urlopen(
            f"https://huggingface.co/{a.repo}/resolve/main/model.safetensors.index.json", timeout=30).read())["weight_map"]
        idx_shards = sorted(set(v for k, v in idx.items()
                                if "indexer" in k and 0 <= layer_idx(k) < a.n_layers))
        tot_gb = len(idx_shards) * 5.4
        print(f"[IDX] indexer weights across {len(idx_shards)} shards (~{tot_gb:.0f} GB total download, resumable)")
        for i, sh in enumerate(idx_shards):
            outp = os.path.join(a.outdir, f"out-idx-{i:05d}.safetensors")
            if os.path.exists(outp): continue             # already done -> resumable
            print(f"[IDX {i+1}/{len(idx_shards)}] downloading {sh}...", flush=True)
            p = download_retry(a.repo, sh, tmp)
            out = {}; convert_shard(p, out, a.n_layers, a.ebits, a.io_bits, a.xbits, keep_idx=True, group_size=a.group_size, bits_map=bits_map)
            if out: save_file(out, outp)
            os.remove(p)
            for blob in glob.glob(os.path.join(tmp, "**", "*"), recursive=True):
                if os.path.isfile(blob): os.remove(blob)
            print(f"    -> {os.path.basename(outp)} ({len(out)} tensors)", flush=True)
        shutil.rmtree(tmp, ignore_errors=True); print("[IDX] DONE."); return
    for i, sh in enumerate(shards):
        if free_gb(a.outdir) < a.min_free_gb:
            print(f"STOP: free space is below {a.min_free_gb} GB. Free space and rerun to resume."); break
        outp = os.path.join(a.outdir, f"out-{i:05d}.safetensors")
        if os.path.exists(outp): continue                 # already done -> resumable
        print(f"[{i+1}/{len(shards)}] downloading {sh} ({free_gb(a.outdir):.0f} GB free)...", flush=True)
        p = download_retry(a.repo, sh, tmp)
        out = {}; convert_shard(p, out, a.n_layers, a.ebits, a.io_bits, a.xbits, group_size=a.group_size, bits_map=bits_map)
        save_file(out, outp)
        os.remove(p)                                       # <-- delete the fp8 shard right away
        for blob in glob.glob(os.path.join(tmp, "**", "*"), recursive=True):
            if os.path.isfile(blob): os.remove(blob)
        print(f"    -> {os.path.basename(outp)} ({os.path.getsize(outp)/1e9:.2f} GB)", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print("DONE." if i == len(shards)-1 else "INTERRUPTED (rerun to resume).")

if __name__ == "__main__":
    main()
