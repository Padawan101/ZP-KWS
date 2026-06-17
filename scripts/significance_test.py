#!/usr/bin/env python3
"""
Bootstrap EER Significance Test for ZP-KWS — Interspeech 2026

Reads .npz files from extract_scores.py.
Uses pre-computed `score` field (system-specific fusion) for TO-KWS/TB-KWS.

Usage:
    python scripts/significance_test.py \
        --baseline scores/phonmatchnet_hard.npz \
        --proposed scores/p_ukws_hard.npz \
        --baseline_name "PhonMatchNet" \
        --proposed_name "ZP-KWS" \
        --metric TO-KWS

    # Multiple metrics (Bonferroni correction)
    python scripts/significance_test.py \
        --baseline scores/phonmatchnet_hard.npz \
        --proposed scores/p_ukws_hard.npz \
        --metric TO-KWS C-KWS \
        --n_bootstraps 1000

    # Also supports npy directory format
    python scripts/significance_test.py \
        --baseline scores/phonmatchnet/libriphrase_hard/ \
        --proposed scores/zpkws/libriphrase_hard/ \
        --metric TO-KWS
"""

import argparse
import os
import sys
import numpy as np

# Shared utilities
sys.path.insert(0, os.path.dirname(__file__))
from eval_utils import compute_eer_interpolated, filter_by_mode


# ============================================================
# Bootstrap Test
# ============================================================

def bootstrap_eer_significance(
    y_true,
    scores_baseline,
    scores_proposed,
    n_bootstraps=1000,
    seed=42,
    confidence_level=0.95,
):
    """
    Bootstrap significance test for EER difference.

    H0: EER_baseline <= EER_proposed (no improvement).
    H1: EER_baseline > EER_proposed (proposed is better).

    Uses paired resampling (same indices for both systems).
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)

    eer_base = compute_eer_interpolated(y_true, scores_baseline)
    eer_prop = compute_eer_interpolated(y_true, scores_proposed)

    diffs = np.zeros(n_bootstraps)

    for i in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)

        y_boot = y_true[indices]
        s_base_boot = scores_baseline[indices]
        s_prop_boot = scores_proposed[indices]

        eer_b = compute_eer_interpolated(y_boot, s_base_boot)
        eer_p = compute_eer_interpolated(y_boot, s_prop_boot)

        diffs[i] = eer_b - eer_p  # Positive = proposed better

    alpha = 1 - confidence_level
    ci_lower = np.percentile(diffs, 100 * alpha / 2)
    ci_upper = np.percentile(diffs, 100 * (1 - alpha / 2))
    p_value = np.mean(diffs <= 0)

    return {
        'eer_baseline': eer_base,
        'eer_proposed': eer_prop,
        'eer_improvement': eer_base - eer_prop,
        'mean_improvement': float(np.mean(diffs)),
        'std_improvement': float(np.std(diffs)),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'p_value': float(p_value),
        'n_samples': n_samples,
        'n_bootstraps': n_bootstraps,
        'confidence_level': confidence_level,
    }


def apply_bonferroni(results_list, alpha=0.05):
    n_tests = len(results_list)
    corrected_alpha = alpha / n_tests
    significant = [r['p_value'] < corrected_alpha for r in results_list]
    return corrected_alpha, significant


# ============================================================
# Data Loading
# ============================================================

def load_scores(path):
    """Load from .npz or npy directory. Returns unified dict."""
    path = str(path)

    if path.endswith('.npz') or (os.path.isfile(path) and not os.path.isdir(path)):
        data = dict(np.load(path, allow_pickle=True))
        return {
            'score': data.get('score', data.get('P_utt')),  # fallback
            'P_utt': data.get('P_utt', data.get('keyword_scores')),
            'keyword_labels': data['keyword_labels'],
            'speaker_labels': data['speaker_labels'],
            'categories': data['categories'],
        }

    elif os.path.isdir(path):
        kw_scores = np.load(os.path.join(path, 'keyword_scores.npy'))
        sp_scores = np.load(os.path.join(path, 'speaker_scores.npy'))
        return {
            'score': kw_scores * sp_scores,  # legacy: assume multiplicative
            'P_utt': kw_scores,
            'keyword_labels': np.load(os.path.join(path, 'keyword_labels.npy')),
            'speaker_labels': np.load(os.path.join(path, 'speaker_labels.npy')),
            'categories': np.load(os.path.join(path, 'categories.npy'), allow_pickle=True),
        }

    else:
        raise FileNotFoundError(f"Score path not found: {path}")


def get_scores_for_mode(data, metric):
    """Get appropriate scores and labels for the given metric mode."""
    if metric == 'C-KWS':
        input_scores = data['P_utt']
    else:
        input_scores = data['score']

    return filter_by_mode(
        input_scores,
        data['keyword_labels'],
        data['speaker_labels'],
        data['categories'],
        metric,
    )


# ============================================================
# Output Formatting
# ============================================================

def format_results(result, baseline_name, proposed_name, dataset_name, metric):
    eer_b = result['eer_baseline'] * 100
    eer_p = result['eer_proposed'] * 100
    imp = result['eer_improvement'] * 100
    rel = (result['eer_improvement'] / result['eer_baseline'] * 100
           if result['eer_baseline'] > 0 else 0)
    ci_lo = result['ci_lower'] * 100
    ci_hi = result['ci_upper'] * 100
    p = result['p_value']

    lines = [
        "=" * 64,
        "Bootstrap EER Significance Test",
        "=" * 64,
        f"Dataset:    {dataset_name} (N={result['n_samples']:,})",
        f"Metric:     {metric} EER",
        f"Bootstraps: {result['n_bootstraps']:,} (seed=42)",
        "",
        f"Baseline ({baseline_name:15s}): {eer_b:.2f}%",
        f"Proposed ({proposed_name:15s}): {eer_p:.2f}%",
        f"Improvement:            {imp:+.2f}% (relative: {rel:.1f}%)",
        "",
        f"95% CI:  [{ci_lo:+.2f}%, {ci_hi:+.2f}%]",
    ]

    p_str = "< 0.001" if p < 0.001 else f"= {p:.3f}"
    lines.append(f"p-value: {p_str}")

    if p < 0.001:
        verdict = "✅ Highly Significant (p < 0.001)"
    elif p < 0.01:
        verdict = "✅ Significant (p < 0.01)"
    elif p < 0.05:
        verdict = "✅ Significant (p < 0.05)"
    else:
        verdict = "❌ Not Significant (p ≥ 0.05)"

    lines.append(f"Result:  {verdict}")
    lines.append("=" * 64)
    return "\n".join(lines)


def generate_latex_snippet(result, baseline_name, proposed_name):
    imp = result['eer_improvement'] * 100
    ci_lo = result['ci_lower'] * 100
    ci_hi = result['ci_upper'] * 100
    p = result['p_value']
    n = result['n_bootstraps']

    p_tex = "$p < 0.001$" if p < 0.001 else (
        "$p < 0.01$" if p < 0.01 else f"$p = {p:.3f}$"
    )

    return (
        f"\n--- LaTeX snippet ---\n"
        f"The improvement of {proposed_name} over {baseline_name} "
        f"({imp:.2f}\\% absolute EER reduction) "
        f"is statistically significant with {p_tex} "
        f"(95\\% CI: [{ci_lo:.2f}\\%, {ci_hi:.2f}\\%], "
        f"bootstrap test with {n:,} resamples).\n"
    )


def save_results_to_file(output_path, all_text, all_results, metrics):
    """Save all metrics' results to file."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(all_text)
        f.write("\n\n--- Raw Values ---\n")
        for metric, result in zip(metrics, all_results):
            f.write(f"\n[{metric}]\n")
            for key in ['eer_baseline', 'eer_proposed', 'eer_improvement',
                         'mean_improvement', 'std_improvement',
                         'ci_lower', 'ci_upper', 'p_value',
                         'n_samples', 'n_bootstraps', 'confidence_level']:
                val = result[key]
                if isinstance(val, float) and key not in ('p_value', 'confidence_level'):
                    f.write(f"  {key}: {val*100:.4f}%\n")
                else:
                    f.write(f"  {key}: {val}\n")
    print(f">> Saved to {output_path}")


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Bootstrap EER Significance Test")
    parser.add_argument('--baseline', type=str, required=True)
    parser.add_argument('--proposed', type=str, required=True)
    parser.add_argument('--baseline_name', type=str, default='Baseline')
    parser.add_argument('--proposed_name', type=str, default='Proposed')
    parser.add_argument('--dataset_name', type=str, default='LibriPhrase Hard')
    parser.add_argument('--metric', type=str, nargs='+', default=['TO-KWS'],
                        choices=['C-KWS', 'TB-KWS', 'TO-KWS'])
    parser.add_argument('--n_bootstraps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--output', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 64)
    print("Bootstrap EER Significance Test")
    print(f"  Baseline: {args.baseline}")
    print(f"  Proposed: {args.proposed}")
    print(f"  Metrics:  {args.metric}")
    print(f"  Bootstraps: {args.n_bootstraps}")
    print("=" * 64 + "\n")

    baseline_data = load_scores(args.baseline)
    proposed_data = load_scores(args.proposed)

    # Verify sample alignment
    assert np.array_equal(
        baseline_data['keyword_labels'],
        proposed_data['keyword_labels']
    ), "ERROR: keyword_labels mismatch! All models must use the same dataloader."

    assert np.array_equal(
        baseline_data['speaker_labels'],
        proposed_data['speaker_labels']
    ), "ERROR: speaker_labels mismatch! All models must use the same speaker pairing."

    print(f">> Labels verified: {len(baseline_data['keyword_labels']):,} samples ✓\n")

    all_results = []
    all_text = []

    for metric in args.metric:
        print(f"--- {metric} ---")

        scores_base, labels = get_scores_for_mode(baseline_data, metric)
        scores_prop, _      = get_scores_for_mode(proposed_data, metric)

        result = bootstrap_eer_significance(
            labels, scores_base, scores_prop,
            n_bootstraps=args.n_bootstraps,
            seed=args.seed,
        )
        all_results.append(result)

        text = format_results(
            result, args.baseline_name, args.proposed_name,
            args.dataset_name, metric,
        )
        print(text)
        print(generate_latex_snippet(result, args.baseline_name, args.proposed_name))
        all_text.append(text)

    # Bonferroni
    if len(args.metric) > 1:
        corr_alpha, significant = apply_bonferroni(all_results, alpha=args.alpha)
        bonf_header = f"\n{'='*64}\nBonferroni (K={len(args.metric)}, α_corrected={corr_alpha:.4f})\n{'='*64}"
        print(bonf_header)
        bonf_lines = [bonf_header]
        for m, sig, res in zip(args.metric, significant, all_results):
            status = "✅ Significant" if sig else "❌ Not Significant"
            line = f"  {m}: p={res['p_value']:.4f} → {status}"
            print(line)
            bonf_lines.append(line)
        all_text.append("\n".join(bonf_lines))

    if args.output:
        save_results_to_file(args.output, "\n\n".join(all_text), all_results, args.metric)


if __name__ == '__main__':
    main()
