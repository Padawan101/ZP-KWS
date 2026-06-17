"""
Score Distribution Diagnostics for Personalized KWS

用法 1: 在 train_personalized.py 中呼叫
    from score_diagnostics import dump_score_diagnostics
    
    # 在 evaluate_extended_dataset 回傳 metrics_calculator 後：
    dump_score_diagnostics(metrics_calculator, dataset_name, epoch, output_dir)

用法 2: 獨立執行（需要先在 validation 時存好 metrics_calculator）
    python score_diagnostics.py --scores_dir results/B0_xxx/score_dumps/ --epoch 5
"""

import numpy as np
import os
import json
from typing import Dict, Optional
from pathlib import Path


def dump_score_diagnostics(
    metrics_calculator,
    dataset_name: str,
    epoch: int,
    output_dir: str,
    save_raw: bool = False,
    fusion_mode: str = 'multiply',
):
    """
    分析並輸出每個 category 的 P_utt / P_spk / Score 分佈統計。
    
    Args:
        metrics_calculator: PersonalizedKWSMetrics instance (已 update 完所有 batch)
        dataset_name: 'Easy', 'Hard', 'GSC', 'Qualcomm' 等
        epoch: 當前 epoch
        output_dir: 輸出目錄
        save_raw: 是否存原始分數（用於後續繪圖）
        fusion_mode: 融合方式
    """
    if len(metrics_calculator.keyword_scores) == 0:
        return
    
    # 合併所有 batch
    kw_scores = np.concatenate(metrics_calculator.keyword_scores)    # P_utt
    spk_scores = np.concatenate(metrics_calculator.speaker_scores)   # P_spk
    kw_labels = np.concatenate(metrics_calculator.keyword_labels)
    spk_labels = np.concatenate(metrics_calculator.speaker_labels)
    categories = np.array(metrics_calculator.categories)
    
    # 計算 fusion score
    if fusion_mode == 'multiply':
        fused = kw_scores * spk_scores
    elif fusion_mode == 'harmonic':
        fused = 2.0 * kw_scores * spk_scores / (kw_scores + spk_scores + 1e-8)
    elif fusion_mode == 'min':
        fused = np.minimum(kw_scores, spk_scores)
    else:
        fused = kw_scores * spk_scores

    # ===============================
    # 1. 每個 category 的統計
    # ===============================
    cat_order = ['ts-tk', 'nts-tk', 'ts-ntk', 'nts-ntk']
    report_lines = []
    report_lines.append(f"{'='*80}")
    report_lines.append(f"Score Diagnostics: {dataset_name} | Epoch {epoch} | Fusion: {fusion_mode}")
    report_lines.append(f"Total samples: {len(kw_scores)}")
    report_lines.append(f"{'='*80}")
    
    stats_dict = {}
    
    for cat in cat_order:
        mask = categories == cat
        n = mask.sum()
        if n == 0:
            continue
        
        p_utt = kw_scores[mask]
        p_spk = spk_scores[mask]
        score = fused[mask]
        
        stats = {
            'count': int(n),
            'P_utt': _dist_stats(p_utt),
            'P_spk': _dist_stats(p_spk),
            'Score': _dist_stats(score),
        }
        stats_dict[cat] = stats
        
        report_lines.append(f"\n--- {cat} (n={n}) ---")
        report_lines.append(f"  {'':>10}  {'mean':>8}  {'std':>8}  {'min':>8}  {'p5':>8}  {'median':>8}  {'p95':>8}  {'max':>8}  {'>0.5':>8}  {'>0.9':>8}")
        for name, arr in [('P_utt', p_utt), ('P_spk', p_spk), ('Score', score)]:
            s = _dist_stats(arr)
            report_lines.append(
                f"  {name:>10}  {s['mean']:8.4f}  {s['std']:8.4f}  {s['min']:8.4f}  "
                f"{s['p5']:8.4f}  {s['median']:8.4f}  {s['p95']:8.4f}  {s['max']:8.4f}  "
                f"{s['frac_gt_0.5']:8.4f}  {s['frac_gt_0.9']:8.4f}"
            )
    
    # ===============================
    # 2. 關鍵診斷指標
    # ===============================
    report_lines.append(f"\n{'='*80}")
    report_lines.append("Key Diagnostic Indicators:")
    report_lines.append(f"{'='*80}")
    
    if 'ts-tk' in stats_dict and 'nts-tk' in stats_dict:
        # TO-KWS 的核心困難：ts-tk Score vs nts-tk Score 的分離度
        ts_tk_score = fused[categories == 'ts-tk']
        nts_tk_score = fused[categories == 'nts-tk']
        
        overlap = _distribution_overlap(ts_tk_score, nts_tk_score)
        report_lines.append(f"  ts-tk vs nts-tk Score overlap ratio:  {overlap:.4f}")
        report_lines.append(f"  (越接近 0 越好，>0.3 表示嚴重重疊)")
        
        # P_utt 在 nts-tk 上的表現 → 如果太高表示 P_utt overconfident
        nts_tk_putt = kw_scores[categories == 'nts-tk']
        report_lines.append(f"\n  nts-tk P_utt > 0.9 比例:  {(nts_tk_putt > 0.9).mean():.4f}")
        report_lines.append(f"  nts-tk P_utt > 0.95 比例: {(nts_tk_putt > 0.95).mean():.4f}")
        report_lines.append(f"  nts-tk P_utt > 0.99 比例: {(nts_tk_putt > 0.99).mean():.4f}")
        report_lines.append(f"  → 如果 >0.9 比例高，表示 P_utt 對正確關鍵詞幾乎無法區分 speaker")
    
    if 'ts-ntk' in stats_dict:
        # ts-ntk 的 P_utt 應該要很低（非目標詞）
        ts_ntk_putt = kw_scores[categories == 'ts-ntk']
        ts_ntk_score = fused[categories == 'ts-ntk']
        report_lines.append(f"\n  ts-ntk P_utt > 0.3 比例:  {(ts_ntk_putt > 0.3).mean():.4f}")
        report_lines.append(f"  ts-ntk P_utt > 0.5 比例:  {(ts_ntk_putt > 0.5).mean():.4f}")
        report_lines.append(f"  ts-ntk Score > 0.3 比例:   {(ts_ntk_score > 0.3).mean():.4f}")
        report_lines.append(f"  → 如果 P_utt > 0.3 比例高，表示 domain shift 導致 KWS 對非目標詞 overconfident")
    
    # ===============================
    # 3. TO-KWS vs SV 天花板分析
    # ===============================
    report_lines.append(f"\n{'='*80}")
    report_lines.append("TO-KWS vs SV Ceiling Analysis:")
    report_lines.append(f"{'='*80}")
    
    # 計算 "如果 P_utt 全部是 1" 的理論 TO-KWS
    theoretical_tokws_scores = spk_scores.copy()  # 全部 P_utt=1 → Score = P_spk
    to_labels = (categories == 'ts-tk').astype(float)
    
    from sklearn.metrics import roc_curve
    
    # 實際 TO-KWS
    fpr_actual, tpr_actual, _ = roc_curve(to_labels, fused)
    fnr_actual = 1 - tpr_actual
    eer_idx = np.nanargmin(np.abs(fpr_actual - fnr_actual))
    eer_actual = (fpr_actual[eer_idx] + fnr_actual[eer_idx]) / 2
    
    # 理論天花板（P_utt=1）
    fpr_ceil, tpr_ceil, _ = roc_curve(to_labels, theoretical_tokws_scores)
    fnr_ceil = 1 - tpr_ceil
    eer_idx_ceil = np.nanargmin(np.abs(fpr_ceil - fnr_ceil))
    eer_ceiling = (fpr_ceil[eer_idx_ceil] + fnr_ceil[eer_idx_ceil]) / 2
    
    # 純 SV
    sv_labels = spk_labels
    if len(np.unique(sv_labels)) >= 2:
        fpr_sv, tpr_sv, _ = roc_curve(sv_labels, spk_scores)
        fnr_sv = 1 - tpr_sv
        eer_idx_sv = np.nanargmin(np.abs(fpr_sv - fnr_sv))
        eer_sv = (fpr_sv[eer_idx_sv] + fnr_sv[eer_idx_sv]) / 2
    else:
        eer_sv = float('nan')
    
    # 純 C-KWS
    fpr_ckws, tpr_ckws, _ = roc_curve(kw_labels, kw_scores)
    fnr_ckws = 1 - tpr_ckws
    eer_idx_ckws = np.nanargmin(np.abs(fpr_ckws - fnr_ckws))
    eer_ckws = (fpr_ckws[eer_idx_ckws] + fnr_ckws[eer_idx_ckws]) / 2
    
    report_lines.append(f"  C-KWS EER (P_utt only):     {eer_ckws*100:.2f}%")
    report_lines.append(f"  SV EER (P_spk only):         {eer_sv*100:.2f}%")
    report_lines.append(f"  TO-KWS EER (actual fusion):  {eer_actual*100:.2f}%")
    report_lines.append(f"  TO-KWS EER (P_utt=1 ceiling):{eer_ceiling*100:.2f}%")
    report_lines.append(f"")
    report_lines.append(f"  Gap: actual - ceiling = {(eer_actual - eer_ceiling)*100:+.2f}%")
    report_lines.append(f"  Gap: ceiling - SV      = {(eer_ceiling - eer_sv)*100:+.2f}%")
    report_lines.append(f"")
    report_lines.append(f"  解讀:")
    report_lines.append(f"    - ceiling ≈ SV → TO-KWS 天花板確實是 SV 品質")
    report_lines.append(f"    - actual ≈ ceiling → P_utt 已飽和，改善 fusion 無用")
    report_lines.append(f"    - actual > ceiling → P_utt 有些 variance 還在幫忙（或在傷害）")
    report_lines.append(f"    - ceiling > SV → ts-ntk 的 P_utt 不夠低，造成額外混淆")
    
    # ===============================
    # 輸出
    # ===============================
    report_text = '\n'.join(report_lines)
    
    # 印到 console
    print(report_text)
    
    # 存到檔案
    diag_dir = os.path.join(output_dir, 'diagnostics')
    os.makedirs(diag_dir, exist_ok=True)
    
    report_path = os.path.join(diag_dir, f'{dataset_name}_epoch{epoch}.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    # 存 JSON（方便後續自動分析）
    json_path = os.path.join(diag_dir, f'{dataset_name}_epoch{epoch}.json')
    json_data = {
        'dataset': dataset_name,
        'epoch': epoch,
        'fusion_mode': fusion_mode,
        'total_samples': len(kw_scores),
        'categories': stats_dict,
        'eer': {
            'C-KWS': float(eer_ckws),
            'SV': float(eer_sv),
            'TO-KWS_actual': float(eer_actual),
            'TO-KWS_ceiling': float(eer_ceiling),
        }
    }
    # Convert numpy types for JSON serialization
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=_json_default)
    
    # 選擇性存原始分數
    if save_raw:
        raw_path = os.path.join(diag_dir, f'{dataset_name}_epoch{epoch}_raw.npz')
        np.savez_compressed(
            raw_path,
            P_utt=kw_scores,
            P_spk=spk_scores,
            Score=fused,
            kw_labels=kw_labels,
            spk_labels=spk_labels,
            categories=categories,
        )
    
    return json_data


def _dist_stats(arr: np.ndarray) -> dict:
    """計算分佈統計"""
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'median': float(np.median(arr)),
        'p5': float(np.percentile(arr, 5)),
        'p95': float(np.percentile(arr, 95)),
        'frac_gt_0.5': float((arr > 0.5).mean()),
        'frac_gt_0.9': float((arr > 0.9).mean()),
    }


def _distribution_overlap(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """
    估算兩個分佈的重疊比例 (0=完全分離, 1=完全重疊)
    用 histogram intersection 方法
    """
    bins = np.linspace(0, 1, 101)
    hist_pos, _ = np.histogram(pos_scores, bins=bins, density=True)
    hist_neg, _ = np.histogram(neg_scores, bins=bins, density=True)
    # Normalize to sum to 1
    hist_pos = hist_pos / (hist_pos.sum() + 1e-10)
    hist_neg = hist_neg / (hist_neg.sum() + 1e-10)
    overlap = np.minimum(hist_pos, hist_neg).sum()
    return float(overlap)


def _json_default(obj):
    """JSON serializer for numpy types"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
