"""
DET Curve Plotting for Interspeech 2026

Read .npz score files (from extract_scores.py) and generate DET curves.
Each .npz contains a pre-computed `score` field (system-specific TO-KWS fusion).

Usage:
    python plot_det.py \
        --scores "PhonMatchNet:results/scores/phonmatchnet_hard.npz" \
                 "PK-MTL:results/scores/pkmtl_hard.npz" \
                 "ZP-KWS (Ours):results/scores/p_ukws_hard.npz" \
        --output figures/det_hard_tokws.pdf \
        --metric tokws
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import det_curve

# Shared utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
from eval_utils import compute_eer_interpolated, filter_by_mode


# ============================================================
# Config
# ============================================================

# Wong (2011) colorblind-safe palette
SYSTEM_STYLES = {
    0: {'color': '#0072B2', 'linestyle': '--'},       # Blue, dashed
    1: {'color': '#D55E00', 'linestyle': '-.'},       # Orange, dashdot
    2: {'color': '#009E73', 'linestyle': '-'},         # Green, solid
    3: {'color': '#CC79A7', 'linestyle': ':'},         # Pink, dotted
    4: {'color': '#F0E442', 'linestyle': (0, (3, 1, 1, 1))},  # Yellow
}

RCPARAMS = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 9,
    'legend.fontsize': 7,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'lines.linewidth': 1.2,
    'figure.dpi': 300,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    'grid.linewidth': 0.3,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot DET curves")
    parser.add_argument('--scores', nargs='+', required=True,
                        help='label:path pairs, e.g. "PhonMatchNet:scores/phon.npz"')
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--metric', type=str, default='tokws',
                        choices=['tokws', 'ckws', 'tbkws'])
    parser.add_argument('--figsize', nargs=2, type=float, default=[3.25, 2.8])
    return parser.parse_args()


# ============================================================
# Score Preparation
# ============================================================

def get_scores_and_labels(data, metric):
    """
    Get filtered scores and labels for a given metric mode.

    Args:
        data: dict from .npz (score, P_utt, keyword_labels, speaker_labels, categories)
        metric: 'tokws', 'ckws', or 'tbkws'

    Returns:
        scores, labels
    """
    mode_map = {'tokws': 'TO-KWS', 'ckws': 'C-KWS', 'tbkws': 'TB-KWS'}
    mode = mode_map[metric]

    if mode == 'C-KWS':
        # C-KWS uses P_utt only
        input_scores = data['P_utt']
    else:
        # TO-KWS / TB-KWS use the pre-computed system score
        input_scores = data['score']

    return filter_by_mode(
        input_scores,
        data['keyword_labels'],
        data['speaker_labels'],
        data['categories'],
        mode,
    )


# ============================================================
# Plotting
# ============================================================

def plot_det(systems, metric, output_path, figsize):
    """Plot DET curves for multiple systems."""
    plt.rcParams.update(RCPARAMS)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Collect all FPR/FNR/EER for auto-ranging
    all_fpr_ranges = []
    all_fnr_ranges = []
    all_eer_pcts = []

    for i, (label, npz_path) in enumerate(systems):
        data = dict(np.load(npz_path, allow_pickle=True))
        scores, labels = get_scores_and_labels(data, metric)

        if len(np.unique(labels)) < 2:
            print(f"WARNING: {label} has < 2 classes for metric={metric}, skipping")
            continue

        fpr, fnr, thresholds = det_curve(labels, scores)
        eer_val = compute_eer_interpolated(labels, scores)
        all_eer_pcts.append(eer_val * 100)

        style = SYSTEM_STYLES.get(i, SYSTEM_STYLES[0])

        ax.plot(
            fpr * 100, fnr * 100,
            color=style['color'],
            linestyle=style['linestyle'],
            label=f"{label} (EER={eer_val*100:.1f}%)",
            zorder=3,
        )

        # Mark EER point
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        ax.plot(
            fpr[eer_idx] * 100, fnr[eer_idx] * 100,
            marker='o', markersize=4,
            color=style['color'],
            zorder=4,
        )

        # Track ranges for auto-scaling
        valid = (fpr > 0) & (fnr > 0)
        if valid.any():
            all_fpr_ranges.extend([fpr[valid].min(), fpr[valid].max()])
            all_fnr_ranges.extend([fnr[valid].min(), fnr[valid].max()])

        print(f"  {label}: EER={eer_val*100:.2f}%, "
              f"N={len(labels)}, pos={labels.sum():.0f}, neg={len(labels)-labels.sum():.0f}")

    # EER reference line
    diag = np.logspace(-1, 2, 100)
    ax.plot(diag, diag, 'k--', linewidth=0.5, alpha=0.3, zorder=1)

    ax.set_xscale('log')
    ax.set_yscale('log')

    # Auto-range: anchor on EER points so curves appear centered
    if all_eer_pcts:
        eer_lo = min(all_eer_pcts)
        eer_hi = max(all_eer_pcts)
        # Show ~0.3x below the best EER to ~1.8x above the worst EER
        lo = max(0.5, eer_lo * 0.3)
        hi = min(70, eer_hi * 1.8)
    elif all_fpr_ranges and all_fnr_ranges:
        lo = max(1, min(min(all_fpr_ranges) * 100, min(all_fnr_ranges) * 100) * 0.8)
        hi = min(70, max(max(all_fpr_ranges) * 100, max(all_fnr_ranges) * 100) * 1.2)
    else:
        lo, hi = 1, 50

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    metric_titles = {'tokws': 'TO-KWS', 'ckws': 'C-KWS', 'tbkws': 'TB-KWS'}
    ax.set_xlabel('False Alarm Rate (%)')
    ax.set_ylabel('Miss Rate (%)')
    #ax.set_title(f'DET Curve — {metric_titles[metric]}', fontsize=9, pad=6)

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
        axis.get_major_formatter().set_scientific(False)

    tick_vals = [v for v in [0.5, 1, 2, 5, 10, 20, 50] if lo <= v <= hi]
    if tick_vals:
        ax.set_xticks(tick_vals)
        ax.set_xticklabels([str(v) for v in tick_vals])
        ax.set_yticks(tick_vals)
        ax.set_yticklabels([str(v) for v in tick_vals])

    ax.grid(True, which='major', alpha=0.3, linewidth=0.3)
    ax.grid(True, which='minor', alpha=0.15, linewidth=0.2)

    ax.legend(loc='upper right', framealpha=0.8, edgecolor='none')

    fig.tight_layout(pad=0.5)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0.02)
    print(f"\n>> Saved to {output_path}")
    plt.close(fig)


def main():
    args = parse_args()

    print("=" * 60)
    print(f"DET Curve: metric={args.metric}")
    print("=" * 60)

    systems = []
    for s in args.scores:
        if ':' not in s:
            raise ValueError(f"Invalid format '{s}', expected 'label:path'")
        label, path = s.split(':', 1)
        systems.append((label, path))
        print(f"  {label}: {path}")
    print()

    plot_det(systems, args.metric, args.output, tuple(args.figsize))


if __name__ == "__main__":
    main()