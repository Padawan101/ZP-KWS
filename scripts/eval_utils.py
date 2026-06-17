"""
Shared utilities for DET curve and significance test scripts.

- Interpolated EER computation
- Score/label filtering by KWS mode
"""

import numpy as np
from sklearn.metrics import roc_curve


def compute_eer_interpolated(y_true, y_score):
    """
    Compute EER using linear interpolation to find exact FPR=FNR crossing.

    More precise than discrete argmin(|FNR - FPR|) approximation.

    Args:
        y_true: np.array [N], binary labels (0/1)
        y_score: np.array [N], prediction scores

    Returns:
        eer: float, Equal Error Rate
    """
    if len(np.unique(y_true)) < 2:
        return float('nan')

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr

    # Find sign change in (FNR - FPR)
    diff = fnr - fpr
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_changes) == 0:
        idx = np.nanargmin(np.abs(diff))
        return (fpr[idx] + fnr[idx]) / 2.0

    # Linear interpolation at first sign change
    i = sign_changes[0]
    d_fpr = fpr[i + 1] - fpr[i]
    d_fnr = fnr[i + 1] - fnr[i]

    denom = d_fpr - d_fnr
    if denom == 0:
        return (fpr[i] + fnr[i]) / 2.0

    t = (fnr[i] - fpr[i]) / denom
    eer = fpr[i] + t * d_fpr

    return float(eer)


def filter_by_mode(scores, keyword_labels, speaker_labels, categories, mode):
    """
    Filter and compute final scores/labels based on KWS evaluation mode.

    For tokws/tbkws, `scores` should already be the system's final fused score.

    Args:
        scores:         [N] final system score (TO-KWS score)
        keyword_labels: [N] binary keyword labels
        speaker_labels: [N] binary speaker labels
        categories:     [N] str array
        mode:           'C-KWS', 'TB-KWS', or 'TO-KWS'

    Returns:
        filtered_scores: [N'] float array
        filtered_labels: [N'] binary array
    """
    categories = np.array(categories)

    if mode == 'C-KWS':
        # C-KWS: score vs keyword_label (all samples)
        # Note: for C-KWS, caller should pass P_utt as scores
        return scores.copy(), keyword_labels.astype(float).copy()

    elif mode == 'TB-KWS':
        # TB-KWS: exclude nts-tk (non-target speaker, target keyword)
        # Positive: ts-tk, Negative: ts-ntk + nts-ntk
        mask = categories != 'nts-tk'
        tb_labels = (categories[mask] == 'ts-tk').astype(float)
        return scores[mask].copy(), tb_labels

    elif mode == 'TO-KWS':
        # TO-KWS: all samples
        # Positive: keyword AND speaker match
        final_labels = ((keyword_labels == 1) & (speaker_labels == 1)).astype(float)
        return scores.copy(), final_labels

    else:
        raise ValueError(f"Unknown mode: {mode}")
