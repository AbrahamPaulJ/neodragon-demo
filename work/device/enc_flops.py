"""MAC count for the collapsed 2-D encoder, and what the naive T=1 export would cost."""
import sys, types
from pathlib import Path
import torch, torch.nn as nn
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work" / "export"))
sys.path.insert(0, str(ROOT / "src" / "neodragon"))
p = types.ModuleType("neodragon"); p.__path__=[str(ROOT/"src"/"neodragon"/"neodragon")]
sys.modules["neodragon"] = p
from export_vae_encoder_2d import Encoder2D
from neodragon.asymmetric_causal_video_vae import AsymmetricCausalVideoVAE

vae = AsymmetricCausalVideoVAE.from_pretrained(ROOT/"work"/"models"/"neodragon"/"causal_video_vae", torch_dtype=torch.float32).eval()
net = Encoder2D(vae).eval()
tot = {"conv": 0, "attn": 0, "gemm": 0}
rows = []
def hk(name):
    def f(m, i, o):
        if isinstance(m, nn.Conv2d):
            mac = o.numel() * m.in_channels // m.groups * m.kernel_size[0] * m.kernel_size[1]
            tot["conv"] += mac
            rows.append((name, tuple(i[0].shape), tuple(o.shape), mac))
        elif isinstance(m, nn.Linear):
            tot["gemm"] += o.numel() * m.in_features
    return f
hs = [m.register_forward_hook(hk(n)) for n, m in net.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
with torch.no_grad():
    net(torch.zeros(1,3,320,512))
for h in hs: h.remove()
# spatial self-attention: 2 * N^2 * C
N, C = 40*64, 512
tot["attn"] = 2 * N * N * C
print("top conv layers by MAC:")
for r in sorted(rows, key=lambda r: -r[3])[:8]:
    print("  {:<40} {} -> {}  {:.2f} GMAC".format(r[0], r[1], r[2], r[3]/1e9))
print()
for k, v in tot.items(): print("  {:<6} {:8.2f} GMAC".format(k, v/1e9))
print("  TOTAL  {:8.2f} GMAC   (naive T=1 5-D export would be ~3x on the conv part: {:.1f})".format(
    sum(tot.values())/1e9, (tot["conv"]*3 + tot["attn"] + tot["gemm"])/1e9))
print("  decoder for reference: 205.07 GMAC at 100.18 ms")
