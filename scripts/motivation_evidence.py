"""Why the project is needed: day vs. night evidence for distant (small) and overlapping (occluded) vehicles.

Uses only the BASELINE's saved results (Person 2, section 12 of yolov10_baseline.ipynb) plus Person 1's annotations,
so it needs no GPU and no images:
  python scripts/motivation_evidence.py --baseline-dir <Drive>/DATA255/outputs/yolov10_baseline
Writes outputs/figures/motivation_day_vs_night.{png,csv}.

Panels: (a) detection and (b) counting are worse at night; (c) the night gap by vehicle size and (d) by occlusion;
(e) small vehicles are common in night TEST videos but rare in night TRAIN/VAL videos; (f) fine-tuning on our data
lowered night recall (zero-shot vs. fine-tuned).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DAY, NIGHT = 'goldenrod', 'midnightblue'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline-dir', required=True, help='folder with condition_breakdown.csv, recall_by_size_occlusion.csv, count_by_occlusion.csv')
    ap.add_argument('--annotations', default=str(ROOT / 'data' / 'processed' / 'annotations.csv'))
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'figures'))
    a = ap.parse_args()
    b, out = Path(a.baseline_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cond = pd.read_csv(b / 'condition_breakdown.csv').set_index(['group', 'model'])
    so = pd.read_csv(b / 'recall_by_size_occlusion.csv').set_index(['group', 'attribute', 'level'])
    occ = pd.read_csv(b / 'count_by_occlusion.csv')
    occ = occ[occ['model'] == 'fine_tuned'].set_index(['group', 'occlusion_group'])
    ann = pd.read_csv(a.annotations, usecols=['split', 'condition', 'vehicle_size', 'occlusion_level'])
    ann['light'] = np.where(ann['condition'] == 'night', 'low_light', 'day')
    test = ann[ann['split'] == 'test']
    n_test = lambda col, lvl, light: int(((test[col] == lvl) & (test['light'] == light)).sum())

    rows = []
    def add(section, item, d, n, n_day=None, n_night=None):
        rows.append({'evidence': section, 'item': item, 'day': d, 'low_light': n, 'gap': n - d, 'n_day': n_day, 'n_low_light': n_night})

    ft = lambda g, m: float(cond.loc[(g, 'fine_tuned'), m])
    for m in ['recall', 'mAP50', 'count_MAE', 'count_error_pct']:
        add('overall (fine-tuned baseline, class ignored)', m, ft('day', m), ft('low_light', m),
            int(cond.loc[('day', 'fine_tuned'), 'vehicles']), int(cond.loc[('low_light', 'fine_tuned'), 'vehicles']))
    SIZES, OCCS = ['small', 'medium', 'large'], ['none', 'partial', 'heavy']
    for attr, levels, col in [('vehicle_size', SIZES, 'vehicle_size'), ('occlusion_level', OCCS, 'occlusion_level')]:
        for lvl in levels:
            add(f'recall by {col}', lvl, float(so.loc[('day', attr, lvl), 'recall_fine_tuned']),
                float(so.loc[('low_light', attr, lvl), 'recall_fine_tuned']), n_test(col, lvl, 'day'), n_test(col, lvl, 'low_light'))
    OCC_GROUPS = ['< 25% occluded', '25-50% occluded', '>= 50% occluded']
    for g in OCC_GROUPS:
        add('count error % by share of occluded vehicles in the frame', g, float(occ.loc[('day', g), 'count_error_pct']),
            float(occ.loc[('low_light', g), 'count_error_pct']), int(occ.loc[('day', g), 'images']), int(occ.loc[('low_light', g), 'images']))
    share = (ann.assign(small=ann['vehicle_size'] == 'small').groupby(['split', 'light'])['small'].mean() * 100).unstack('light')
    for s in ['train', 'val', 'test']:
        add('% of ground-truth boxes that are small (<32 px)', s, share.loc[s, 'day'], share.loc[s, 'low_light'])
    for m in ['zero_shot', 'fine_tuned']:
        add('recall before vs after fine-tuning', m, float(cond.loc[('day', m), 'recall']), float(cond.loc[('low_light', m), 'recall']))
    table = pd.DataFrame(rows)
    table.to_csv(out / 'motivation_day_vs_night.csv', index=False)

    fig, ax = plt.subplots(2, 3, figsize=(16, 8.6))

    def pair(axis, labels, d, n, title, ylabel, ylim=None, fmt='{:.2f}', ns=None):
        x = np.arange(len(labels)); w = 0.38
        b1 = axis.bar(x - w / 2, d, w, color=DAY, label='day')
        b2 = axis.bar(x + w / 2, n, w, color=NIGHT, label='night (low light)')
        for bars, vals in [(b1, d), (b2, n)]:
            for r, v in zip(bars, vals):
                axis.text(r.get_x() + r.get_width() / 2, r.get_height(), fmt.format(v), ha='center', va='bottom', fontsize=8.5)
        axis.set_xticks(x)
        axis.set_xticklabels(labels if ns is None else [f'{l}\n(n={a:,} / {b:,})' for l, (a, b) in zip(labels, ns)], fontsize=9)
        axis.set_title(title, fontsize=11, loc='left'); axis.set_ylabel(ylabel)
        if ylim: axis.set_ylim(*ylim)
        axis.grid(axis='y', alpha=0.3); axis.set_axisbelow(True)

    t = table.set_index(['evidence', 'item'])
    sec = lambda s, items: (t.loc[[(s, i) for i in items], 'day'].to_numpy(), t.loc[[(s, i) for i in items], 'low_light'].to_numpy(),
                            list(zip(t.loc[[(s, i) for i in items], 'n_day'].fillna(0).astype(int), t.loc[[(s, i) for i in items], 'n_low_light'].fillna(0).astype(int))))
    s0 = 'overall (fine-tuned baseline, class ignored)'
    d, n, _ = sec(s0, ['recall', 'mAP50'])
    pair(ax[0, 0], ['vehicle recall', 'vehicle mAP50'], d, n, '(a) Detection is worse at night', 'score', (0, 1.08))
    d, n, _ = sec(s0, ['count_error_pct'])
    d2, n2, ns = sec('count error % by share of occluded vehicles in the frame', OCC_GROUPS)
    pair(ax[0, 1], ['all frames'] + [g.replace(' occluded', '') + '\noccluded' for g in OCC_GROUPS], np.r_[d, d2], np.r_[n, n2],
         '(b) Counting error is 2-3x larger at night,\n     most when many vehicles overlap', 'count error (% of true count)', (0, 50), '{:.0f}')
    d, n, ns = sec('recall by vehicle_size', SIZES)
    pair(ax[0, 2], ['small\n<32 px', 'medium\n32-96 px', 'large\n>96 px'], d, n, '(c) Distant (small) vehicles: worst at night',
         'recall', (0, 1.12), ns=ns)
    d, n, ns = sec('recall by occlusion_level', OCCS)
    pair(ax[1, 0], ['none', 'partial\n(<half covered)', 'heavy\n(>=half covered)'], d, n, '(d) Overlapping vehicles: partly covered\n     ones are missed at night', 'recall', (0, 1.12), ns=ns)
    d, n, _ = sec('% of ground-truth boxes that are small (<32 px)', ['train', 'val', 'test'])
    pair(ax[1, 1], ['train', 'val', 'test'], d, n, '(e) Small vehicles at night are common in TEST\n     but rare in TRAIN/VAL', '% of boxes that are small', (0, 36), '{:.1f}')
    d, n, _ = sec('recall before vs after fine-tuning', ['zero_shot', 'fine_tuned'])
    pair(ax[1, 2], ['pretrained (zero-shot)', 'fine-tuned on our data'], d, n, '(f) Fine-tuning lowered night recall', 'vehicle recall', (0, 1.08))
    ax[0, 0].legend(loc='lower left', fontsize=9)
    fig.suptitle('YOLOv10-S baseline, UA-DETRAC test set: day vs. night (3 night test videos; n = day / night boxes)', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / 'motivation_day_vs_night.png', dpi=150)
    print(table.round(3).to_string(index=False))
    print('Saved', out / 'motivation_day_vs_night.png')


if __name__ == '__main__':
    main()
