"""Architecture diagram for the TA requirement: YOLOv10-S baseline vs. YOLOv10-S + P2 head.
Reads models/yolov10s-p2.yaml and checks it has the structure drawn here before drawing.
  python scripts/draw_architecture.py      -> outputs/figures/architecture_baseline_vs_p2.png
Channel counts are for the 's' scale (width 0.5) and grid sizes for a 640x640 input."""
from pathlib import Path
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / 'models' / 'yolov10s-p2.yaml'))
head = cfg['head']
# layer index = 11 + position in head
L = {11 + i: h for i, h in enumerate(head)}
assert L[23][0] == 16 and L[23][2] == 'nn.Upsample', 'P2 branch should upsample layer 16 (P3)'
assert L[24][0] == [-1, 2] and L[24][2] == 'Concat', 'P2 branch should concat backbone layer 2 (stride 4)'
assert L[25][2] == 'C2f' and L[25][3] == [128], 'P2 branch should end in C2f(128 -> 64 channels at width 0.5)'
assert L[26][0] == [25, 16, 19, 22] and L[26][2] == 'v10Detect', 'head should read P2, P3, P4, P5'
assert len(cfg['backbone']) == 11, 'backbone should be the unchanged 11 YOLOv10-S layers'

OLD, NEW, HEAD, EDGE = 'lightsteelblue', 'sandybrown', 'lightgray', 'darkslategray'
fig, ax = plt.subplots(figsize=(16.6, 9.8))
ax.set_xlim(0, 16.6); ax.set_ylim(-2.5, 10.6); ax.axis('off')


def box(x, y, w, h, text, color, bold=False, fs=9.5, hatch=None):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
                                fc=color, ec=EDGE, lw=1.6 if bold else 1.0, hatch=hatch))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, weight='bold' if bold else 'normal', linespacing=1.35)


def arrow(p, q, color=EDGE, rad=0.0, lw=1.3, ls='-'):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=12, color=color, lw=lw, ls=ls,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=2, shrinkB=2))


# rows = stride 4 (top) ... stride 32 (bottom)
Y = {4: 8.0, 8: 5.8, 16: 3.6, 32: 1.4}
XB, XT, XU, XH = 2.0, 6.0, 9.6, 13.4
W, H = 3.1, 1.35

for x, t in [(XB, 'BACKBONE (unchanged)'), (XT, 'NECK: top-down'), (XU, 'NECK: bottom-up'), (XH, 'DETECTION HEAD (v10Detect)')]:
    ax.text(x, 10.15, t, ha='center', fontsize=11, weight='bold', color=EDGE)
box(XB, 9.45, 2.4, 0.5, 'input 640 x 640 x 3', 'white', fs=9)
box(XB, Y[4], W, H, 'layers 0-2: Conv, Conv, C2f\nstride 4 -> 160x160x64', OLD)
box(XB, Y[8], W, H, 'layers 3-4: Conv, C2f\nstride 8 -> 80x80x128', OLD)
box(XB, Y[16], W, H, 'layers 5-6: SCDown, C2f\nstride 16 -> 40x40x256', OLD)
box(XB, Y[32], W, H, 'layers 7-10: SCDown, C2fCIB,\nSPPF, PSA (attention)\nstride 32 -> 20x20x512', OLD)
for a, b in [(9.2, Y[4] + H / 2), (Y[4] - H / 2, Y[8] + H / 2), (Y[8] - H / 2, Y[16] + H / 2), (Y[16] - H / 2, Y[32] + H / 2)]:
    arrow((XB, a), (XB, b))

box(XT, Y[4], W, H, 'NEW layers 23-25:\nupsample P3, concat layer 2, C2f\n-> P2 160x160x64', NEW, bold=True)
box(XT, Y[8], W, H, 'layers 14-16: upsample,\nconcat layer 4, C2f\n-> P3 80x80x128', OLD)
box(XT, Y[16], W, H, 'layers 11-13: upsample,\nconcat layer 6, C2f\n-> 40x40x256', OLD)
box(XU, Y[16], W, H, 'layers 17-19: Conv (stride 2),\nconcat layer 13, C2f\n-> P4 40x40x256', OLD)
box(XU, Y[32], W, H, 'layers 20-22: SCDown,\nconcat layer 10, C2fCIB\n-> P5 20x20x512', OLD)

# lateral connections from the backbone
arrow((XB + W / 2, Y[4]), (XT - W / 2, Y[4]), color='peru', lw=2.2)
arrow((XB + W / 2, Y[8]), (XT - W / 2, Y[8]))
arrow((XB + W / 2, Y[16]), (XT - W / 2, Y[16]))
arrow((XB + W / 2, Y[32]), (XT - W / 2 + 0.2, Y[16] - H / 2), rad=0.15)          # layer 10 -> 11 (upsample)
arrow((XB + W / 2, Y[32] - 0.35), (XU - W / 2, Y[32] - 0.35))                     # layer 10 -> 21 (concat)
# top-down
arrow((XT, Y[16] + H / 2), (XT, Y[8] - H / 2))
arrow((XT, Y[8] + H / 2), (XT, Y[4] - H / 2), color='peru', lw=2.2)
# bottom-up
arrow((XT + W / 2, Y[8] - 0.3), (XU - W / 2, Y[16] + 0.3), rad=-0.1)              # P3 -> 17
arrow((XT + W / 2, Y[16]), (XU - W / 2, Y[16]))                                   # 13 -> 18
arrow((XU, Y[16] - H / 2), (XU, Y[32] + H / 2))

# heads
HW = 3.0
box(XH, Y[4], HW, H, 'NEW scale P2: stride 4\n160 x 160 grid (4 px cells)\nvery small / distant vehicles', NEW, bold=True)
box(XH, Y[8], HW, H, 'P3: stride 8\n80 x 80 grid (8 px cells)\nsmall vehicles', HEAD)
box(XH, Y[16], HW, H, 'P4: stride 16\n40 x 40 grid (16 px cells)\nmedium vehicles', HEAD)
box(XH, Y[32], HW, H, 'P5: stride 32\n20 x 20 grid (32 px cells)\nlarge vehicles', HEAD)
arrow((XT + W / 2, Y[4]), (XH - HW / 2, Y[4]), color='peru', lw=2.2)
arrow((XT + W / 2, Y[8] + 0.25), (XH - HW / 2, Y[8] + 0.25))
arrow((XU + W / 2, Y[16]), (XH - HW / 2, Y[16]))
arrow((XU + W / 2, Y[32]), (XH - HW / 2, Y[32]))
ax.text(XH, 0.25, 'Each scale predicts boxes + 4 classes (car, bus, van, others).\n'
        'YOLOv10 head: one-to-one branch, so no NMS is needed.', ha='center', va='center', fontsize=9, style='italic')

# legend
box(1.3, -0.45, 0.5, 0.35, '', OLD); ax.text(1.7, -0.45, 'same as baseline: loads the pretrained yolov10s.pt weights (layers 0-22)', va='center', fontsize=9.5)
box(1.3, -0.95, 0.5, 0.35, '', NEW); ax.text(1.7, -0.95, 'added by us: P2 branch (layers 23-25) and 4th head scale, start untrained', va='center', fontsize=9.5)
box(1.3, -1.45, 0.5, 0.35, '', HEAD); ax.text(1.7, -1.45, 'detection head (layer 26): rebuilt for 4 scales, so it also starts untrained in the P2 model', va='center', fontsize=9.5)

rows = [('', 'Baseline YOLOv10-S', 'Ours: YOLOv10-S + P2'),
        ('detection scales', 'P3, P4, P5 (strides 8 / 16 / 32)', 'P2, P3, P4, P5 (strides 4 / 8 / 16 / 32)'),
        ('finest grid at 640 input', '80 x 80 (8 px cells)', '160 x 160 (4 px cells)'),
        ('layers', '24 (0-23)', '27 (0-26): +3 neck layers, 4-input head'),
        ('training', 'pretrained yolov10s.pt -> fine-tuned', 'same weights, data, split and settings')]
for r, (a, b, c) in enumerate(rows):
    y = -0.45 - 0.3 * r
    kw = dict(fontsize=9.5, va='center', weight='bold' if r == 0 else 'normal')
    ax.text(8.3, y, a, **kw); ax.text(10.45, y, b, **kw); ax.text(13.35, y, c, **kw)
ax.plot([8.3, 16.5], [-0.62, -0.62], color=EDGE, lw=0.6)
ax.text(1.0, -2.15, 'Parameters and GFLOPs: from yolov10_p2.ipynb, section 3 (architecture check).',
        fontsize=8.5, style='italic', va='center')
fig.suptitle('YOLOv10-S + P2 head: a fourth, finer detection scale for small and distant vehicles', fontsize=14, weight='bold', y=0.985)
fig.tight_layout()
out = ROOT / 'outputs' / 'figures' / 'architecture_baseline_vs_p2.png'
fig.savefig(out, dpi=150)
print('Saved', out)
