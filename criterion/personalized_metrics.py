"""
Evaluation Metrics for Personalized Keyword Spotting

Implements metrics for three evaluation scenarios:
- C-KWS (Conventional KWS): Standard keyword verification
- TB-KWS (Target-Biased KWS): Preference for target speaker
- TO-KWS (Target-Only KWS): Strict personalization

Metrics:
- EER (Equal Error Rate)
- AUC (Area Under Curve)
- FRR @ FAR (False Rejection Rate at fixed False Alarm Rate)
- FAR @ FRR (False Alarm Rate at fixed False Rejection Rate)

FIXED: matplotlib is now optional (only imported when plotting)
"""

import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
from typing import Dict, List, Tuple, Optional
from pathlib import Path
# matplotlib moved to plot_roc_curves() - now optional


class PersonalizedKWSMetrics:
    """
    Metrics calculator for personalized keyword spotting

    Supports three evaluation modes:
    - C-KWS: Conventional (keyword verification only)
    - TB-KWS: Target-Biased (prefer target speaker)
    - TO-KWS: Target-Only (strict personalization)
    """

    def __init__(self, fusion_mode: str = 'multiply'):
        """Initialize metrics calculator
        
        Args:
            fusion_mode: Score fusion strategy ('multiply', 'harmonic', 'min')
        """
        assert fusion_mode in ('multiply', 'harmonic', 'min'), \
            f"fusion_mode must be 'multiply', 'harmonic', or 'min', got {fusion_mode}"
        self.fusion_mode = fusion_mode
        self.reset()

    def reset(self):
        """Reset all stored predictions and labels"""
        self.keyword_scores = []  # P_utt
        self.speaker_scores = []  # P_spk
        self.keyword_labels = []
        self.speaker_labels = []
        self.categories = []

    def update(
        self,
        keyword_scores: torch.Tensor,
        speaker_scores: torch.Tensor,
        keyword_labels: torch.Tensor,
        speaker_labels: torch.Tensor,
        categories: List[str]
    ):
        """
        Add batch of predictions

        Args:
            keyword_scores: Keyword match scores (P_utt) [B]
            speaker_scores: Speaker match scores (P_spk) [B]
            keyword_labels: Keyword match labels [B] (1=match, 0=mismatch)
            speaker_labels: Speaker match labels [B] (1=same, 0=different)
            categories: List of categories ['ts-tk', 'nts-tk', ...]
        """
        # Convert to numpy
        if isinstance(keyword_scores, torch.Tensor):
            keyword_scores = keyword_scores.cpu().numpy()
        if isinstance(speaker_scores, torch.Tensor):
            speaker_scores = speaker_scores.cpu().numpy()
        if isinstance(keyword_labels, torch.Tensor):
            keyword_labels = keyword_labels.cpu().numpy()
        if isinstance(speaker_labels, torch.Tensor):
            speaker_labels = speaker_labels.cpu().numpy()

        # Ensure 1D
        keyword_scores = keyword_scores.flatten()
        speaker_scores = speaker_scores.flatten()
        keyword_labels = keyword_labels.flatten()
        speaker_labels = speaker_labels.flatten()

        # Store
        self.keyword_scores.append(keyword_scores)
        self.speaker_scores.append(speaker_scores)
        self.keyword_labels.append(keyword_labels)
        self.speaker_labels.append(speaker_labels)
        self.categories.extend(categories)

    def _fuse_scores(self, kw_scores: np.ndarray, spk_scores: np.ndarray) -> np.ndarray:
        """Fuse keyword and speaker scores using configured fusion mode."""
        if self.fusion_mode == 'multiply':
            return kw_scores * spk_scores
        elif self.fusion_mode == 'harmonic':
            return 2.0 * kw_scores * spk_scores / (kw_scores + spk_scores + 1e-8)
        elif self.fusion_mode == 'min':
            return np.minimum(kw_scores, spk_scores)

    def compute(self) -> Dict[str, Dict[str, float]]:
        """
        Compute all metrics for all three modes

        Returns:
            results: Dict with keys 'C-KWS', 'TB-KWS', 'TO-KWS', 'SV'
                     Each containing EER, AUC, FRR@FAR, etc.
        """
        if len(self.keyword_scores) == 0:
            return {}

        # Concatenate all batches
        kw_scores = np.concatenate(self.keyword_scores)  # P_utt
        spk_scores = np.concatenate(self.speaker_scores)  # P_spk
        kw_labels = np.concatenate(self.keyword_labels)
        spk_labels = np.concatenate(self.speaker_labels)
        categories = np.array(self.categories)

        results = {}

        # === C-KWS Metrics ===
        # Use ONLY keyword scores (P_utt)
        # Positive: keyword match (regardless of speaker)
        # Negative: keyword mismatch
        c_kws_preds = kw_scores  # Use P_utt only
        c_kws_labels = kw_labels
        results['C-KWS'] = self._compute_metrics(c_kws_preds, c_kws_labels, mode='C-KWS')

        # === TB-KWS Metrics ===
        # Use combined scores (P_utt × P_spk)
        # Positive: ts-tk (target speaker + target keyword)
        # Negative: ts-ntk + nts-ntk
        # Neutral (excluded): nts-tk
        combined_scores = self._fuse_scores(kw_scores, spk_scores)
        tb_mask = categories != 'nts-tk'
        tb_labels = (categories == 'ts-tk').astype(float)
        if tb_mask.sum() > 0:
            results['TB-KWS'] = self._compute_metrics(
                combined_scores[tb_mask],
                tb_labels[tb_mask],
                mode='TB-KWS'
            )
        else:
            results['TB-KWS'] = {'error': 'No samples for TB-KWS'}

        # === TO-KWS Metrics ===
        # Use combined scores (P_utt × P_spk)
        # Positive: ts-tk only
        # Negative: nts-tk + ts-ntk + nts-ntk
        to_labels = (categories == 'ts-tk').astype(float)
        results['TO-KWS'] = self._compute_metrics(combined_scores, to_labels, mode='TO-KWS')

        # === Speaker Verification Metrics ===
        # Use ONLY speaker scores (P_spk)
        # Positive: target speaker (ts-tk + ts-ntk)
        # Negative: non-target speaker (nts-tk + nts-ntk)
        sv_preds = spk_scores  # Use P_spk only
        sv_labels = spk_labels
        results['SV'] = self._compute_metrics(sv_preds, sv_labels, mode='SV')

        return results

    def _compute_metrics(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        mode: str = ''
    ) -> Dict[str, float]:
        """
        Compute metrics for a single task

        Args:
            preds: Predictions [N]
            labels: Ground truth labels [N]
            mode: Task mode (for logging)

        Returns:
            metrics: Dict with EER, AUC, FRR@FAR, etc.
        """
        # Check if we have both classes
        if len(np.unique(labels)) < 2:
            return {
                'error': 'Not enough classes',
                'n_samples': len(labels),
                'n_positive': int(labels.sum()),
                'n_negative': int((1 - labels).sum()),
            }

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(labels, preds)
        fnr = 1 - tpr

        # === EER (Equal Error Rate) ===
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]

        # === AUC (Area Under Curve) ===
        roc_auc = auc(fpr, tpr)

        # === FRR @ fixed FAR ===
        frr_at_far = {}
        for target_far in [0.01, 0.05, 0.10]:
            # Find closest FAR point
            idx = np.searchsorted(fpr, target_far)
            if idx < len(fnr):
                frr_value = fnr[idx]
            else:
                frr_value = fnr[-1]
            frr_at_far[f'FRR@FAR{int(target_far*100)}%'] = frr_value

        # === FAR @ fixed FRR ===
        far_at_frr = {}
        for target_frr in [0.01, 0.05]:
            # Find closest FRR point (search from high to low)
            idx = np.searchsorted(fnr[::-1], target_frr)
            idx = len(fnr) - 1 - idx
            if idx >= 0 and idx < len(fpr):
                far_value = fpr[idx]
            else:
                far_value = fpr[0]
            far_at_frr[f'FAR@FRR{int(target_frr*100)}%'] = far_value

        # Assemble results
        metrics = {
            'EER': float(eer),
            'EER_threshold': float(eer_threshold),
            'AUC': float(roc_auc),
            'n_samples': len(labels),
            'n_positive': int(labels.sum()),
            'n_negative': int((1 - labels).sum()),
            **{k: float(v) for k, v in frr_at_far.items()},
            **{k: float(v) for k, v in far_at_frr.items()},
        }

        return metrics

    def get_category_breakdown(self) -> Dict[str, int]:
        """
        Get sample counts per category

        Returns:
            counts: Dict mapping category to count
        """
        if len(self.categories) == 0:
            return {}

        categories = np.array(self.categories)
        unique, counts = np.unique(categories, return_counts=True)

        return dict(zip(unique, counts))

    def get_diagnostic_report(self, mode: str = 'TO-KWS') -> str:
        """
        Generate comprehensive diagnostic report

        Args:
            mode: Primary mode for error analysis ('TO-KWS' recommended)

        Returns:
            report: Formatted diagnostic report string
        """
        if len(self.keyword_scores) == 0:
            return "No data collected for diagnostics."

        # Concatenate all batches
        kw_scores = np.concatenate(self.keyword_scores)  # P_utt
        spk_scores = np.concatenate(self.speaker_scores)  # P_spk
        kw_labels = np.concatenate(self.keyword_labels)
        spk_labels = np.concatenate(self.speaker_labels)
        categories = np.array(self.categories)
        combined_scores = self._fuse_scores(kw_scores, spk_scores)

        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("P-PhonMatchNet Diagnostic Report")
        lines.append("=" * 80)

        # === 1. Category Score Statistics ===
        lines.append("\n[1] Category Score Statistics")
        lines.append("-" * 60)

        cat_stats = {}
        for cat in ['ts-tk', 'nts-tk', 'ts-ntk', 'nts-ntk']:
            mask = categories == cat
            if mask.sum() > 0:
                cat_stats[cat] = {
                    'P_utt': kw_scores[mask],
                    'P_spk': spk_scores[mask],
                    'score': combined_scores[mask],
                    'count': mask.sum()
                }
                # Calculate statistics for P_utt
                putt_mean = np.mean(kw_scores[mask])
                putt_std = np.std(kw_scores[mask])
                putt_median = np.median(kw_scores[mask])
                putt_min = np.min(kw_scores[mask])
                putt_max = np.max(kw_scores[mask])
                
                # Calculate statistics for P_spk
                pspk_mean = np.mean(spk_scores[mask])
                pspk_std = np.std(spk_scores[mask])
                pspk_median = np.median(spk_scores[mask])
                pspk_min = np.min(spk_scores[mask])
                pspk_max = np.max(spk_scores[mask])
                
                # Calculate statistics for score
                score_mean = np.mean(combined_scores[mask])
                score_std = np.std(combined_scores[mask])
                score_median = np.median(combined_scores[mask])
                score_min = np.min(combined_scores[mask])
                score_max = np.max(combined_scores[mask])
                
                # Format output with complete statistics
                lines.append(f"  {cat:8s} (N={mask.sum()}):")
                lines.append(f"    P_utt: mean={putt_mean:.3f}, std={putt_std:.3f}, "
                           f"median={putt_median:.3f}, range=[{putt_min:.3f}, {putt_max:.3f}]")
                lines.append(f"    P_spk: mean={pspk_mean:.3f}, std={pspk_std:.3f}, "
                           f"median={pspk_median:.3f}, range=[{pspk_min:.3f}, {pspk_max:.3f}]")
                lines.append(f"    score: mean={score_mean:.3f}, std={score_std:.3f}, "
                           f"median={score_median:.3f}, range=[{score_min:.3f}, {score_max:.3f}]")
            else:
                lines.append(f"  {cat:8s}: No samples")

        # === 2. Independent Branch EER ===
        lines.append("\n[2] Independent Branch EER")
        lines.append("-" * 60)

        # Keyword EER
        if len(np.unique(kw_labels)) >= 2:
            fpr, tpr, _ = roc_curve(kw_labels, kw_scores)
            fnr = 1 - tpr
            eer_idx = np.nanargmin(np.abs(fpr - fnr))
            kw_eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
            lines.append(f"  Keyword EER:  {kw_eer*100:.2f}%")
        else:
            lines.append(f"  Keyword EER:  N/A (insufficient class variety)")

        # Speaker EER
        if len(np.unique(spk_labels)) >= 2:
            fpr, tpr, _ = roc_curve(spk_labels, spk_scores)
            fnr = 1 - tpr
            eer_idx = np.nanargmin(np.abs(fpr - fnr))
            spk_eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
            lines.append(f"  Speaker EER:  {spk_eer*100:.2f}%")
        else:
            lines.append(f"  Speaker EER:  N/A (insufficient class variety)")

        # === 3. Error Analysis (at EER threshold) ===
        lines.append(f"\n[3] {mode} Error Analysis (at EER threshold)")
        lines.append("-" * 60)

        # Compute threshold for mode
        if mode == 'TO-KWS':
            labels = (categories == 'ts-tk').astype(float)
            preds = combined_scores
        elif mode == 'TB-KWS':
            tb_mask = categories != 'nts-tk'
            labels = (categories[tb_mask] == 'ts-tk').astype(float)
            preds = combined_scores[tb_mask]
            categories_subset = categories[tb_mask]
        else:
            labels = kw_labels
            preds = kw_scores

        if len(np.unique(labels)) >= 2:
            fpr, tpr, thresholds = roc_curve(labels, preds)
            fnr = 1 - tpr
            eer_idx = np.nanargmin(np.abs(fpr - fnr))
            threshold = thresholds[eer_idx]

            # Find FA and FR samples
            if mode == 'TO-KWS':
                fa_mask = (combined_scores > threshold) & (labels == 0)
                fr_mask = (combined_scores <= threshold) & (labels == 1)
            else:
                fa_mask = (preds > threshold) & (labels == 0)
                fr_mask = (preds <= threshold) & (labels == 1)

            # FA analysis
            fa_count = fa_mask.sum()
            lines.append(f"  EER Threshold: {threshold:.4f}")
            lines.append(f"  False Accepts: {fa_count}")
            
            if fa_count > 0 and mode == 'TO-KWS':
                fa_cats = categories[fa_mask]
                for cat in ['nts-tk', 'ts-ntk', 'nts-ntk']:
                    cat_mask = (fa_cats == cat)
                    cat_count = cat_mask.sum()
                    if cat_count > 0:
                        avg_putt = np.mean(kw_scores[fa_mask][cat_mask])
                        avg_pspk = np.mean(spk_scores[fa_mask][cat_mask])
                        lines.append(f"    - from {cat:8s}: {cat_count:5d} "
                                   f"(avg P_utt={avg_putt:.3f}, avg P_spk={avg_pspk:.3f})")

            # FR analysis
            fr_count = fr_mask.sum()
            lines.append(f"  False Rejects: {fr_count}")
            if fr_count > 0:
                avg_putt = np.mean(kw_scores[fr_mask])
                avg_pspk = np.mean(spk_scores[fr_mask])
                lines.append(f"    - avg P_utt={avg_putt:.3f}, avg P_spk={avg_pspk:.3f}")
        else:
            lines.append("  N/A (insufficient class variety)")

        # === 4. Diagnosis ===
        lines.append("\n[4] Diagnosis")
        lines.append("-" * 60)

        # Analyze bottleneck
        bottleneck = "Unknown"
        suggestion = "Collect more data"

        if 'ts-tk' in cat_stats and 'nts-tk' in cat_stats:
            nts_tk_pspk_mean = np.mean(cat_stats['nts-tk']['P_spk'])
            ts_tk_pspk_mean = np.mean(cat_stats['ts-tk']['P_spk'])
            
            # Check if SV branch is the bottleneck
            if nts_tk_pspk_mean > 0.5:
                bottleneck = "SV Branch (P_spk too high for non-target speakers)"
                suggestion = "Adjust spk_bias or try Cross-Attention"
            elif spk_eer > 0.15:
                bottleneck = "SV Branch (high Speaker EER)"
                suggestion = "Fine-tune Speaker Encoder or try Cross-Attention"
            elif nts_tk_pspk_mean < 0.3 and fa_count > 0:
                # Check if P_utt is too high for nts-tk
                nts_tk_putt_mean = np.mean(cat_stats['nts-tk']['P_utt'])
                if nts_tk_putt_mean > 0.7:
                    bottleneck = "Fusion Strategy (P_utt too high overwhelms low P_spk)"
                    suggestion = "Try Learnable Fusion or weighted combination"
                else:
                    bottleneck = "Fusion Strategy"
                    suggestion = "Try Learnable Fusion"
            else:
                bottleneck = "Balanced (no clear bottleneck)"
                suggestion = "Continue training or try ensemble"

        lines.append(f"  主要瓶頸: {bottleneck}")
        lines.append(f"  建議方向: {suggestion}")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)

    def plot_roc_curves(self, save_path: Optional[str] = None):
        """
        Plot ROC curves for all three modes

        Args:
            save_path: Path to save figure (if None, just display)
        """
        # Import matplotlib only when needed
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("ERROR: matplotlib not installed. Install with: pip install matplotlib")
            print("       Skipping ROC curve plotting...")
            return

        if len(self.predictions) == 0:
            print("No data to plot")
            return

        # Concatenate data
        preds = np.concatenate(self.predictions)
        kw_labels = np.concatenate(self.keyword_labels)
        spk_labels = np.concatenate(self.speaker_labels)
        categories = np.array(self.categories)

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        # Plot 1: C-KWS
        c_kws_labels = kw_labels
        fpr, tpr, _ = roc_curve(c_kws_labels, preds)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.4f}')
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('C-KWS (Conventional)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: TB-KWS
        tb_mask = categories != 'nts-tk'
        tb_labels = (categories == 'ts-tk').astype(float)
        if tb_mask.sum() > 0:
            fpr, tpr, _ = roc_curve(tb_labels[tb_mask], preds[tb_mask])
            roc_auc = auc(fpr, tpr)
            axes[1].plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.4f}')
        axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('TB-KWS (Target-Biased)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot 3: TO-KWS
        to_labels = (categories == 'ts-tk').astype(float)
        fpr, tpr, _ = roc_curve(to_labels, preds)
        roc_auc = auc(fpr, tpr)
        axes[2].plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.4f}')
        axes[2].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[2].set_xlabel('False Positive Rate')
        axes[2].set_ylabel('True Positive Rate')
        axes[2].set_title('TO-KWS (Target-Only)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # Plot 4: Speaker Verification
        sv_labels = spk_labels
        fpr, tpr, _ = roc_curve(sv_labels, preds)
        roc_auc = auc(fpr, tpr)
        axes[3].plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.4f}')
        axes[3].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[3].set_xlabel('False Positive Rate')
        axes[3].set_ylabel('True Positive Rate')
        axes[3].set_title('Speaker Verification')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to {save_path}")
        else:
            plt.show()

        plt.close()


def format_metrics_report(metrics: Dict[str, Dict[str, float]], dataset_name: str = "") -> str:
    """
    Format metrics as a readable report

    Args:
        metrics: Output from PersonalizedKWSMetrics.compute()
        dataset_name: Name of dataset

    Returns:
        report: Formatted string report
    """
    lines = []
    lines.append("=" * 80)
    lines.append("P-PhonMatchNet Evaluation Report")
    lines.append("=" * 80)

    if dataset_name:
        lines.append(f"\nDataset: {dataset_name}")

    # Category breakdown
    if 'C-KWS' in metrics and 'n_samples' in metrics['C-KWS']:
        lines.append(f"Total samples: {metrics['C-KWS']['n_samples']}")

    # C-KWS
    if 'C-KWS' in metrics:
        lines.append("\n" + "-" * 80)
        lines.append("C-KWS (Conventional KWS)")
        lines.append("-" * 80)
        lines.append("Positive: keyword match | Negative: keyword mismatch")
        lines.append("")
        m = metrics['C-KWS']
        if 'error' not in m:
            lines.append(f"  EER:              {m['EER']*100:.2f}%")
            lines.append(f"  AUC:              {m['AUC']*100:.2f}%")
            lines.append(f"  FRR @ FAR 1%:     {m['FRR@FAR1%']*100:.2f}%")
            lines.append(f"  FRR @ FAR 5%:     {m['FRR@FAR5%']*100:.2f}%")
            lines.append(f"  FRR @ FAR 10%:    {m['FRR@FAR10%']*100:.2f}%")
        else:
            lines.append(f"  Error: {m['error']}")

    # TB-KWS
    if 'TB-KWS' in metrics:
        lines.append("\n" + "-" * 80)
        lines.append("TB-KWS (Target-Biased KWS)")
        lines.append("-" * 80)
        lines.append("Positive: ts-tk | Negative: ts-ntk + nts-ntk | Neutral: nts-tk")
        lines.append("")
        m = metrics['TB-KWS']
        if 'error' not in m:
            lines.append(f"  EER:              {m['EER']*100:.2f}%")
            lines.append(f"  AUC:              {m['AUC']*100:.2f}%")
            lines.append(f"  FRR @ FAR 1%:     {m['FRR@FAR1%']*100:.2f}%")
            lines.append(f"  FRR @ FAR 5%:     {m['FRR@FAR5%']*100:.2f}%")
            lines.append(f"  FRR @ FAR 10%:    {m['FRR@FAR10%']*100:.2f}%")
        else:
            lines.append(f"  Error: {m['error']}")

    # TO-KWS
    if 'TO-KWS' in metrics:
        lines.append("\n" + "-" * 80)
        lines.append("TO-KWS (Target-Only KWS)")
        lines.append("-" * 80)
        lines.append("Positive: ts-tk | Negative: nts-tk + ts-ntk + nts-ntk")
        lines.append("")
        m = metrics['TO-KWS']
        if 'error' not in m:
            lines.append(f"  EER:              {m['EER']*100:.2f}%")
            lines.append(f"  AUC:              {m['AUC']*100:.2f}%")
            lines.append(f"  FRR @ FAR 1%:     {m['FRR@FAR1%']*100:.2f}%")
            lines.append(f"  FRR @ FAR 5%:     {m['FRR@FAR5%']*100:.2f}%")
            lines.append(f"  FRR @ FAR 10%:    {m['FRR@FAR10%']*100:.2f}%")
        else:
            lines.append(f"  Error: {m['error']}")

    # SV
    if 'SV' in metrics:
        lines.append("\n" + "-" * 80)
        lines.append("SV (Speaker Verification)")
        lines.append("-" * 80)
        lines.append("Positive: target speaker | Negative: non-target speaker")
        lines.append("")
        m = metrics['SV']
        if 'error' not in m:
            lines.append(f"  EER:              {m['EER']*100:.2f}%")
            lines.append(f"  AUC:              {m['AUC']*100:.2f}%")
        else:
            lines.append(f"  Error: {m['error']}")

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


def save_metrics_to_file(metrics: Dict[str, Dict[str, float]], save_path: str, dataset_name: str = ""):
    """
    Save metrics report to file

    Args:
        metrics: Output from PersonalizedKWSMetrics.compute()
        save_path: Path to save report
        dataset_name: Name of dataset
    """
    report = format_metrics_report(metrics, dataset_name)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        f.write(report)

    print(f"Metrics report saved to {save_path}")