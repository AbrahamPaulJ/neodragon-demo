"""Trap #2 audit: the causal-mask constant.

The upstream source never writes a mask fill value -- masks are BOOLEAN
(`==`, `>=`, `&` in PyramidMMDiT.merge_input) and handed straight to
F.scaled_dot_product_attention. The dangerous constant is injected by the
SDPA lowering, so it is invisible to source review. This exports the exact
pattern and reports what actually lands in the graph.
"""

import os
import tempfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import numpy_helper

NUM_HEADS = 24
HEAD_DIM = 64
SEQ = 64  # small; the constant does not depend on shape


class SdpaBoolMask(nn.Module):
    """Mirrors VarlenSelfAttentionWithT5Mask.__call__ inner attention call."""

    def forward(self, q, k, v, mask):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False, attn_mask=mask
        )


def build_mask():
    """Mirrors merge_input: token-id equality AND temporal causal, both bool."""
    token_ids = torch.ones(1, SEQ, dtype=torch.long)
    order_ids = torch.arange(SEQ).reshape(1, SEQ).float()
    same = token_ids[:, None, :, None] == token_ids[:, None, None, :]
    causal = order_ids[:, None, :, None] >= order_ids[:, None, None, :]
    return same & causal


def scan_graph(path):
    model = onnx.load(path)
    hits = []
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype.kind == "f" and arr.size and not np.all(np.isfinite(arr)):
            hits.append((init.name, "initializer", str(arr.dtype), "non-finite", float(np.nanmin(arr))))
        elif arr.dtype.kind == "f" and arr.size and np.abs(arr).max() > 1e30:
            hits.append((init.name, "initializer", str(arr.dtype), "extreme", float(arr.min())))
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.name == "value" and attr.t.ByteSize():
                arr = numpy_helper.to_array(attr.t)
                if arr.dtype.kind == "f" and arr.size:
                    if not np.all(np.isfinite(arr)):
                        hits.append((node.name or node.op_type, node.op_type, str(arr.dtype),
                                     "non-finite", float(np.nanmin(arr))))
                    elif np.abs(arr).max() > 1e30:
                        hits.append((node.name or node.op_type, node.op_type, str(arr.dtype),
                                     "extreme", float(arr.min())))
    ops = sorted({n.op_type for n in model.graph.node})
    return hits, ops


def main():
    print("=" * 78)
    print("TRAP #2 -- causal mask constant, as emitted by ONNX export")
    print("=" * 78)

    q = torch.randn(1, NUM_HEADS, SEQ, HEAD_DIM)
    k = torch.randn(1, NUM_HEADS, SEQ, HEAD_DIM)
    v = torch.randn(1, NUM_HEADS, SEQ, HEAD_DIM)
    mask = build_mask()

    print(f"\n[source] mask dtype from merge_input pattern: {mask.dtype}  shape {tuple(mask.shape)}")
    print("         no fill value appears anywhere in the Python source.")

    ref = SdpaBoolMask()(q, k, v, mask)
    print(f"[eager]  SDPA output finite: {torch.isfinite(ref).all().item()}")

    for opset in (17, 20):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, f"sdpa_{opset}.onnx")
            try:
                torch.onnx.export(
                    SdpaBoolMask(), (q, k, v, mask), path,
                    input_names=["q", "k", "v", "mask"], output_names=["out"],
                    opset_version=opset, dynamo=False,
                )
            except Exception as e:  # noqa: BLE001
                print(f"\n[opset {opset}] export failed: {type(e).__name__}: {e}")
                continue

            hits, ops = scan_graph(path)
            print(f"\n[opset {opset}] ops: {', '.join(ops)}")
            if hits:
                print(f"[opset {opset}] DANGEROUS CONSTANTS FOUND:")
                for name, kind, dt, why, val in hits:
                    print(f"    {why:<10} {val!r:>16}  dtype={dt:<8} in {kind} '{name}'")
            else:
                print(f"[opset {opset}] no extreme constants found")

    print("\n[reference ranges]")
    print(f"    float32 min          {np.finfo(np.float32).min:.6e}")
    print(f"    float16 min          {np.finfo(np.float16).min:.6e}   <-- overflows below this")
    print(f"    a16 (uint16) span    0 .. 65535 quantisation levels")
    print("\n  Any fill at -3.4e38 is unrepresentable in fp16 and destroys the")
    print("  activation scale for a W8A16 quantiser: one outlier sets the range")
    print("  for the whole softmax input tensor.")

    # what a safe replacement looks like
    print("\n[safe replacement] explicit additive mask instead of bool SDPA")
    for fill in (-1e4, -1e3, -100.0, -60.0):
        add = torch.zeros_like(mask, dtype=torch.float32).masked_fill(~mask, fill)
        scores = (q @ k.transpose(-1, -2)) / HEAD_DIM**0.5 + add
        attn = scores.softmax(-1)
        out = attn @ v
        leak = attn.masked_select(~mask.expand_as(attn)).max().item()
        err = (out - ref).abs().max().item()
        print(f"    fill={fill:>9.1f}  max leaked attn weight={leak:.3e}  max|out-ref|={err:.3e}")


if __name__ == "__main__":
    main()
