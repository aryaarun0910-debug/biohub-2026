#!/usr/bin/env python3
"""EXP-11 -- what detector size actually FITS, measured rather than guessed.

Three hard budgets decide this before any training starts:

  INFERENCE  a kernels-only submission has <=12h for ~199 hidden datasets x 100 frames.
             This is the binding constraint and it is pure arithmetic.
  TRAINING   the M5 Pro has 48GB unified; activations for 3D convs dominate, not weights.
  DATA       133,318 annotated cells is the whole supervision budget (2.82% of the corpus).

Builds the architecture detect.py specifies -- Upsample+Conv3d decoders, bias=True everywhere,
a (1,4,4) stem -- at a range of widths and measures params and FLOPs with torch's own counter.
"""
import sys, math
import torch, torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

T, Z, Y, X = 100, 64, 256, 256          # one dataset
N_TEST = 199                            # "hidden test ~ same size as training"
KERNEL_S = 12 * 3600


def blk(cin, cout):                      # bias=True: the MPS fast path needs it
    return nn.Sequential(nn.Conv3d(cin, cout, 3, padding=1, bias=True), nn.ReLU(inplace=True),
                         nn.Conv3d(cout, cout, 3, padding=1, bias=True), nn.ReLU(inplace=True))


class UNet3D(nn.Module):
    """Dual-head: dense node logits + sparse refinement offsets, per detect.py."""
    def __init__(self, c=32, depth=3):
        super().__init__()
        self.stem = nn.Conv3d(1, c, 3, stride=(1, 4, 4), padding=1, bias=True)  # (1,4,4) grid
        self.down, self.pool = nn.ModuleList(), nn.MaxPool3d(2)
        ch = [c * 2 ** i for i in range(depth + 1)]
        for i in range(depth):
            self.down.append(blk(ch[i], ch[i + 1]))
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        for i in range(depth, 0, -1):                    # Upsample+Conv3d, never ConvTranspose3d
            self.up.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                         nn.Conv3d(ch[i], ch[i - 1], 3, padding=1, bias=True)))
            self.dec.append(blk(ch[i - 1] * 2, ch[i - 1]))
        self.node = nn.Conv3d(c, 1, 1, bias=True)        # dense head
        self.refine = nn.Conv3d(c, 3, 1, bias=True)      # sparse head (3 offsets)

    def forward(self, v):
        x = self.stem(v); skips = []
        for d in self.down:
            skips.append(x); x = d(self.pool(x))
        for u, dc, s in zip(self.up, self.dec, reversed(skips)):
            x = dc(torch.cat([u(x), s], 1))
        return self.node(x), self.refine(x)


def measure(c, depth, patch):
    m = UNet3D(c, depth).eval()
    params = sum(p.numel() for p in m.parameters())
    v = torch.zeros(1, 1, *patch)
    with FlopCounterMode(display=False) as f:
        with torch.no_grad(): m(v)
    return params, f.get_total_flops(), math.prod(patch)


PATCH = (64, 256, 256)
print(f"  one dataset = {T}x{Z}x{Y}x{X};  {N_TEST} test datasets = {N_TEST*T:,} volumes")
print(f"  kernel budget {KERNEL_S/3600:.0f}h; allow 8h for detection\n")
print(f"  {'base_c':>7} {'depth':>5} {'params':>12} {'GFLOP/vol':>10} {'TFLOP total':>12} "
      f"{'T4 hrs':>8} {'M5 hrs':>8}")

T4_EFF, M5_EFF = 15e12, 5e12     # realistic 3D-conv throughput, not peak TFLOPS
for c in (8, 16, 24, 32, 48, 64):
    for depth in (3,):
        p, fl, nvox = measure(c, depth, PATCH)
        total = fl * N_TEST * T
        print(f"  {c:>7} {depth:>5} {p:>12,} {fl/1e9:>10.1f} {total/1e12:>12.0f} "
              f"{total/T4_EFF/3600:>8.2f} {total/M5_EFF/3600:>8.2f}")

print("\n  training memory, one patch, fp32 activations (weights are negligible):")
for c in (16, 32, 48, 64):
    m = UNet3D(c, 3)
    act = [0]
    h = []
    def hook(mod, i, o):
        if isinstance(o, torch.Tensor): act[0] += o.numel()
    for mod in m.modules():
        if isinstance(mod, (nn.Conv3d, nn.ReLU, nn.MaxPool3d, nn.Upsample)):
            h.append(mod.register_forward_hook(hook))
    with torch.no_grad(): m(torch.zeros(1, 1, *PATCH))
    for x in h: x.remove()
    act = act[0]
    gb = act * 4 / 1e9
    print(f"    c={c:>3}: {act/1e6:>9.0f} M activations = {gb:>6.2f} GB fwd, "
          f"~{gb*3:>6.2f} GB with grads (batch 1)")
