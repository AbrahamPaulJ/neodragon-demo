"""Phase 1a -- DistilT5 -> ONNX at static shape [1, 128].

Exports twice: once as-is, once with the trap fixes applied, and scans both
graphs for the constants that break a fixed-point runtime. Verifies each
against the PyTorch reference with onnxruntime.

Fixes applied in the patched export:
  * finite additive attention mask instead of the -3.4e38 that
    T5Stack.get_extended_attention_mask injects (trap #2, T5 side);
  * eager attention so the graph contains explicit MatMul/Softmax rather
    than a fused SDPA the converter would have to pattern-match.
"""

import argparse
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "src" / "neodragon"
MODEL = ROOT / "work" / "models" / "neodragon"
OUT = ROOT / "work" / "onnx"
sys.path.insert(0, str(REPO))
_pkg = types.ModuleType("neodragon")
_pkg.__path__ = [str(REPO / "neodragon")]
sys.modules["neodragon"] = _pkg

from neodragon.distil_t5 import T5EncoderWithProjection  # noqa: E402
from transformers import T5Tokenizer  # noqa: E402

SEQ = 128       # text_encoder_bundle.MAX_SEQUENCE_LENGTH
BATCH = 1
MASK_FILL = -100.0   # sized against measured pre-softmax scores, see report_scores()


class Wrapper(nn.Module):
    """Fixed-shape single-output wrapper: (input_ids, attention_mask) -> embeds."""

    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, input_ids, attention_mask):
        return self.enc(input_ids=input_ids, attention_mask=attention_mask,
                        return_dict=False)[0]


def load(patched: bool):
    enc = T5EncoderWithProjection.from_pretrained(
        MODEL / "text_encoder_3", torch_dtype=torch.float32,
        attn_implementation="eager").eval()
    if patched:
        patch_mask(enc)
    return enc


def patch_mask(enc):
    """Replace the -3.4e38 additive mask with a finite one."""
    stack = enc.encoder.encoder

    def finite_mask(attention_mask, input_shape, *a, **k):
        m = attention_mask[:, None, None, :].to(torch.float32)
        return (1.0 - m) * MASK_FILL

    stack.get_extended_attention_mask = finite_mask
    return enc


def report_scores(enc, ids, am):
    """Measure the pre-softmax score range so MASK_FILL can be justified."""
    lo, hi = [], []

    def hook(_m, inp, _out):
        s = inp[0]
        lo.append(s.min().item())
        hi.append(s.max().item())

    hs = []
    for block in enc.encoder.encoder.block:
        sa = block.layer[0].SelfAttention
        for name, mod in sa.named_modules():
            if isinstance(mod, nn.Softmax):
                hs.append(mod.register_forward_hook(hook))
    if not hs:  # transformers uses functional softmax; fall back to manual calc
        with torch.no_grad():
            e = enc.encoder.encoder
            h = e.embed_tokens(ids)
            b0 = e.block[0].layer[0]
            n = b0.layer_norm(h)
            sa = b0.SelfAttention
            q = sa.q(n).view(ids.shape[0], SEQ, sa.n_heads, sa.key_value_proj_dim).transpose(1, 2)
            k = sa.k(n).view(ids.shape[0], SEQ, sa.n_heads, sa.key_value_proj_dim).transpose(1, 2)
            sc = q @ k.transpose(-1, -2)
            pb = sa.compute_bias(SEQ, SEQ)
            tot = sc + pb
        return tot.min().item(), tot.max().item()
    with torch.no_grad():
        enc(input_ids=ids, attention_mask=am)
    for h in hs:
        h.remove()
    return min(lo), max(hi)


def scan(path):
    model = onnx.load(str(path))
    bad = []
    def check(arr, where):
        if arr.dtype.kind != "f" or arr.size == 0:
            return
        if not np.all(np.isfinite(arr)):
            bad.append((where, "non-finite", float(np.nanmin(arr))))
        elif np.abs(arr).max() > 1e30:
            bad.append((where, "extreme", float(arr.min())))
    for init in model.graph.initializer:
        check(numpy_helper.to_array(init), f"initializer {init.name}")
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.name == "value" and attr.t.ByteSize():
                check(numpy_helper.to_array(attr.t), f"{node.op_type} {node.name}")
    ops = {}
    for n in model.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    return bad, ops, model


def export(enc, path, ids, am):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        Wrapper(enc), (ids, am), str(path),
        input_names=["input_ids", "attention_mask"],
        output_names=["prompt_embeds"],
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    return path


def verify(path, ids, am, ref):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input_ids": ids.numpy(), "attention_mask": am.numpy()})[0]
    got = torch.from_numpy(got)
    return (got - ref).abs().max().item(), (got - ref).norm().item() / ref.norm().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a red sports car driving along a coastal road at sunset")
    args = ap.parse_args()

    tok = T5Tokenizer.from_pretrained(MODEL / "tokenizer_3")
    b = tok([args.prompt], padding="max_length", max_length=SEQ, truncation=True,
            add_special_tokens=True, return_tensors="pt")
    ids, am = b.input_ids[:BATCH], b.attention_mask[:BATCH]

    print("=" * 80)
    print(f"PHASE 1a -- DistilT5 -> ONNX  (batch={BATCH}, seq={SEQ})")
    print("=" * 80)

    base = load(patched=False)
    with torch.no_grad():
        ref = Wrapper(base)(ids, am)
    print(f"\n[reference] output {tuple(ref.shape)}  max|.|={ref.abs().max():.4f}")

    smin, smax = report_scores(base, ids, am)
    print(f"[scores] pre-softmax logit range (block 0): [{smin:.2f}, {smax:.2f}]")
    print(f"         MASK_FILL={MASK_FILL} sits {smin - MASK_FILL:.1f} below the minimum "
          f"real score")

    results = {}
    for tag, patched in (("baseline", False), ("patched", True)):
        enc = load(patched=patched)
        p = OUT / f"distilt5_{tag}.onnx"
        export(enc, p, ids, am)
        bad, ops, _ = scan(p)
        mb = p.stat().st_size / 1024**2
        err, rel = verify(p, ids, am, ref)
        results[tag] = (mb, bad, ops, err, rel)

        print(f"\n--- {tag} ---")
        print(f"  file      {p.name}  {mb:.1f} MB")
        print(f"  ops       {len(ops)} types, {sum(ops.values())} nodes")
        print(f"  vs torch  max|diff|={err:.3e}  rel={rel:.3e}")
        if bad:
            print(f"  UNSAFE CONSTANTS ({len(bad)}):")
            for where, why, val in bad[:6]:
                print(f"      {why:<12}{val!r:>16}  in {where}")
        else:
            print("  constants  clean (no non-finite, none above 1e30)")

    print("\n[op histogram, patched]")
    ops = results["patched"][2]
    for k in sorted(ops, key=lambda x: -ops[x]):
        print(f"  {ops[k]:>5}  {k}")


if __name__ == "__main__":
    main()
