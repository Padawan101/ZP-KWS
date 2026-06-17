#!/usr/bin/env python3
"""
P_phon 詳細分析腳本

針對特定 checkpoint 進行深入的 P_phon 分析，
產出統計數據、視覺化圖表、錯誤案例列表。

Usage:
    python analyze_pphon.py \
        --checkpoint checkpoints/epoch_10.pt \
        --val_pkl DB/libriphrase/personalized_libriphrase_lmdb/easy/val.pkl \
        --output_dir analysis_results/

Output:
    analysis_results/
    ├── pphon_stats.json           # 完整統計數據
    ├── pphon_distribution.png     # P_phon 分布直方圖
    ├── pphon_tp_vs_fp.png         # TP vs FP 對比圖
    ├── easy_vs_hard.png           # Easy vs Hard 對比 (if both loaders provided)
    └── error_cases.csv            # 錯誤案例列表
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.manifold import TSNE
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import p_ukws
from dataset import personalized_libriphrase
from dataset import KWSDataLoader


class PPhonAnalyzer:
    """P_phon 詳細分析器"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 數據收集器
        self.data = {
            # P_phon 值
            'pphon_mean': [],      # [N]
            'pphon_min': [],       # [N]
            'pphon_max': [],       # [N]
            'pphon_std': [],       # [N]
            
            # P_utt 值
            'putt': [],            # [N]
            
            # P_spk 值 (NEW: for personalized KWS analysis)
            'pspk': [],            # [N]
            
            # Fusion score (NEW: P_utt × P_spk)
            'score': [],           # [N]
            
            # Labels
            'z_keyword': [],       # [N]
            'z_speaker': [],       # [N]
            'difficulty': [],      # [N] 'easy' or 'hard' (if available)
            
            # Metadata
            'category': [],        # [N] 'ts-tk', 'ts-ntk', etc.
            'audio_path': [],      # [N] audio file paths (if available)
            'keyword_text': [],    # [N] keyword text (if available)
        }
        
        # Track warnings (to avoid repeated messages)
        self._warned_z_keyword = False
        self._warned_z_speaker = False
        
        # NEW: Embedding collection for t-SNE visualization
        self.embeddings = {
            'emb_audio': [],      # [N, feature_dim] audio embeddings
            'emb_text': [],       # [N, feature_dim] text embeddings  
            'emb_attention': [],  # [N, feature_dim] attention output embeddings
        }
        self.collect_embeddings = False  # Will be set True if embeddings are available
    
    def collect(self, outputs, batch, difficulty='unknown'):
        """
        收集一個 batch 的數據
        
        Args:
            outputs: model forward 輸出
            batch: 輸入 batch
            difficulty: 'easy' or 'hard'
        """
        # 從 seq_ce_logit 計算 P_phon
        if 'seq_ce_logit' not in outputs:
            return
            
        seq_ce_logit = outputs['seq_ce_logit']  # [B, T_t]
        seq_ce_logit_mask = outputs.get('seq_ce_logit_mask', torch.ones_like(seq_ce_logit))
        
        # 轉換為 probability
        P_phon = torch.sigmoid(seq_ce_logit)  # [B, T_t]
        
        # 只計算有效位置（mask=1）
        valid_counts = seq_ce_logit_mask.sum(dim=-1).clamp(min=1)  # [B]
        
        # P_phon mean (masked)
        P_phon_masked = P_phon * seq_ce_logit_mask
        pphon_mean = (P_phon_masked.sum(dim=-1) / valid_counts).cpu()  # [B]
        
        # FIX: Correct masked std calculation
        # The old code used P_phon_masked.std() which incorrectly includes masked 0s
        # Correct approach: compute variance using only valid positions
        pphon_mean_expanded = pphon_mean.unsqueeze(-1).to(P_phon.device)  # [B, 1]
        diff_sq = ((P_phon - pphon_mean_expanded) ** 2) * seq_ce_logit_mask  # [B, T_t]
        pphon_var = diff_sq.sum(dim=-1) / valid_counts.clamp(min=1)  # [B]
        pphon_std = torch.sqrt(pphon_var).cpu()  # [B]
        
        # 對於 min/max，需要處理 mask
        P_phon_for_min = P_phon.clone()
        P_phon_for_min[seq_ce_logit_mask == 0] = 1.0  # Set masked to max for min computation
        pphon_min = P_phon_for_min.min(dim=-1).values.cpu()  # [B]
        
        P_phon_for_max = P_phon.clone()
        P_phon_for_max[seq_ce_logit_mask == 0] = 0.0  # Set masked to min for max computation
        pphon_max = P_phon_for_max.max(dim=-1).values.cpu()  # [B]
        
        # P_utt
        P_utt = outputs['P_utt'].squeeze(-1).cpu()  # [B]
        
        # P_spk (NEW: for personalized KWS)
        if 'P_spk' in outputs:
            P_spk = outputs['P_spk'].squeeze(-1).cpu()  # [B]
        else:
            P_spk = torch.ones_like(P_utt)  # Default to 1.0 if not available
        
        # Fusion Score (NEW: P_utt × P_spk)
        score = P_utt * P_spk  # [B]
        
        # 收集
        self.data['pphon_mean'].append(pphon_mean)
        self.data['pphon_min'].append(pphon_min)
        self.data['pphon_max'].append(pphon_max)
        self.data['pphon_std'].append(pphon_std)
        self.data['putt'].append(P_utt)
        self.data['pspk'].append(P_spk)
        self.data['score'].append(score)
        
        # Labels - FIX: Add warnings for fallback key usage
        if 'z_keyword' in batch:
            z_keyword = batch['z_keyword']
        elif 'z' in batch:
            z_keyword = batch['z']
            if not self._warned_z_keyword:
                print("⚠️ Warning: 'z_keyword' not found in batch, using 'z' as fallback")
                self._warned_z_keyword = True
        else:
            z_keyword = torch.zeros(P_utt.shape[0])
            if not self._warned_z_keyword:
                print("⚠️ Warning: Neither 'z_keyword' nor 'z' found in batch, using zeros!")
                self._warned_z_keyword = True
        z_keyword = z_keyword.squeeze(-1).cpu() if z_keyword.dim() > 1 else z_keyword.cpu()
        self.data['z_keyword'].append(z_keyword)
        
        # Speaker label - FIX: Add warning for missing speaker_label
        if 'speaker_label' in batch:
            z_speaker = batch['speaker_label']
        else:
            z_speaker = torch.zeros(P_utt.shape[0])
            if not self._warned_z_speaker:
                print("⚠️ Warning: 'speaker_label' not found in batch, by_speaker analysis will be invalid!")
                self._warned_z_speaker = True
        z_speaker = z_speaker.squeeze(-1).cpu() if z_speaker.dim() > 1 else z_speaker.cpu()
        self.data['z_speaker'].append(z_speaker)
        
        # Difficulty - extract from batch if available (allows single PKL with both easy/hard)
        batch_size = P_utt.shape[0]
        if 'type' in batch:
            # Convert type strings to difficulty labels
            # 'diffspk_easyneg' -> 'easy', 'diffspk_hardneg' -> 'hard'
            for t in batch['type']:
                if isinstance(t, str):
                    if 'easy' in t.lower():
                        self.data['difficulty'].append('easy')
                    elif 'hard' in t.lower():
                        self.data['difficulty'].append('hard')
                    else:
                        self.data['difficulty'].append(difficulty)
                else:
                    self.data['difficulty'].append(difficulty)
        else:
            self.data['difficulty'].extend([difficulty] * batch_size)
        
        # Category
        if 'category' in batch:
            self.data['category'].extend(batch['category'])
        else:
            self.data['category'].extend(['unknown'] * batch_size)
        
        # NEW: Collect audio paths (for error case analysis)
        if 'audio_path' in batch:
            self.data['audio_path'].extend(batch['audio_path'])
        elif 'wav_path' in batch:
            self.data['audio_path'].extend(batch['wav_path'])
        else:
            self.data['audio_path'].extend([''] * batch_size)
        
        # NEW: Collect keyword text (for error case analysis)
        if 'keyword_text' in batch:
            self.data['keyword_text'].extend(batch['keyword_text'])
        elif 'keyword' in batch:
            self.data['keyword_text'].extend(batch['keyword'])
        else:
            self.data['keyword_text'].extend([''] * batch_size)
        
        # NEW: Collect embeddings for t-SNE if available
        if 'emb_audio' in outputs:
            self.collect_embeddings = True
            self.embeddings['emb_audio'].append(outputs['emb_audio'].cpu())
        if 'emb_text' in outputs:
            self.embeddings['emb_text'].append(outputs['emb_text'].cpu())
        if 'emb_attention' in outputs:
            self.embeddings['emb_attention'].append(outputs['emb_attention'].cpu())
    
    def finalize(self):
        """合併所有 batch 的數據"""
        for key in ['pphon_mean', 'pphon_min', 'pphon_max', 'pphon_std', 
                    'putt', 'pspk', 'score', 'z_keyword', 'z_speaker']:
            if self.data[key]:
                self.data[key] = torch.cat(self.data[key]).numpy()
            else:
                self.data[key] = np.array([])
        
        self.data['difficulty'] = np.array(self.data['difficulty'])
        self.data['category'] = np.array(self.data['category'])
        self.data['audio_path'] = np.array(self.data['audio_path'])
        self.data['keyword_text'] = np.array(self.data['keyword_text'])
        
        # NEW: Finalize embeddings
        if self.collect_embeddings:
            for key in ['emb_audio', 'emb_text', 'emb_attention']:
                if self.embeddings[key]:
                    self.embeddings[key] = torch.cat(self.embeddings[key]).numpy()
                else:
                    self.embeddings[key] = np.array([])
            print(f"\n📊 Embeddings collected: audio={self.embeddings['emb_audio'].shape}, "
                  f"text={self.embeddings['emb_text'].shape}, "
                  f"attention={self.embeddings['emb_attention'].shape}")
    
    def compute_statistics(self):
        """計算所有統計數據"""
        stats = {}
        
        z_kw = self.data['z_keyword']
        if len(z_kw) == 0:
            return {'error': 'No data collected'}
            
        pos_mask = z_kw == 1
        neg_mask = z_kw == 0
        
        # === 1. 基本統計 ===
        stats['basic'] = {
            'total_samples': int(len(z_kw)),
            'positive_samples': int(pos_mask.sum()),
            'negative_samples': int(neg_mask.sum()),
            
            # P_phon mean
            'positive_pphon_mean': float(self.data['pphon_mean'][pos_mask].mean()) if pos_mask.any() else 0,
            'positive_pphon_mean_std': float(self.data['pphon_mean'][pos_mask].std()) if pos_mask.any() else 0,
            'negative_pphon_mean': float(self.data['pphon_mean'][neg_mask].mean()) if neg_mask.any() else 0,
            'negative_pphon_mean_std': float(self.data['pphon_mean'][neg_mask].std()) if neg_mask.any() else 0,
            
            # P_phon min
            'positive_pphon_min': float(self.data['pphon_min'][pos_mask].mean()) if pos_mask.any() else 0,
            'positive_pphon_min_std': float(self.data['pphon_min'][pos_mask].std()) if pos_mask.any() else 0,
            'negative_pphon_min': float(self.data['pphon_min'][neg_mask].mean()) if neg_mask.any() else 0,
            'negative_pphon_min_std': float(self.data['pphon_min'][neg_mask].std()) if neg_mask.any() else 0,
            
            # Gaps
            'gap_mean': float(self.data['pphon_mean'][pos_mask].mean() - self.data['pphon_mean'][neg_mask].mean()) if pos_mask.any() and neg_mask.any() else 0,
            'gap_min': float(self.data['pphon_min'][pos_mask].mean() - self.data['pphon_min'][neg_mask].mean()) if pos_mask.any() and neg_mask.any() else 0,
        }
        
        # === 2. 分布分離度 ===
        if pos_mask.any() and neg_mask.any():
            pos_mean = self.data['pphon_mean'][pos_mask]
            neg_mean = self.data['pphon_mean'][neg_mask]
            
            # d-prime
            pooled_std = np.sqrt((pos_mean.std()**2 + neg_mean.std()**2) / 2)
            d_prime = (pos_mean.mean() - neg_mean.mean()) / pooled_std if pooled_std > 0 else 0
            
            # AUC using pphon_mean as score
            all_scores = np.concatenate([pos_mean, neg_mean])
            all_labels = np.concatenate([np.ones(len(pos_mean)), np.zeros(len(neg_mean))])
            auc_pphon_mean = roc_auc_score(all_labels, all_scores)
            
            # AUC using pphon_min as score
            pos_min = self.data['pphon_min'][pos_mask]
            neg_min = self.data['pphon_min'][neg_mask]
            all_scores_min = np.concatenate([pos_min, neg_min])
            auc_pphon_min = roc_auc_score(all_labels, all_scores_min)
            
            # === NEW: AUC using P_utt ===
            pos_putt = self.data['putt'][pos_mask]
            neg_putt = self.data['putt'][neg_mask]
            all_putt = np.concatenate([pos_putt, neg_putt])
            auc_putt = roc_auc_score(all_labels, all_putt)
            
            # === NEW: AUC using P_utt × P_phon_min (combined feature) ===
            pos_combined = pos_putt * pos_min
            neg_combined = neg_putt * neg_min
            all_combined = np.concatenate([pos_combined, neg_combined])
            auc_combined = roc_auc_score(all_labels, all_combined)
            
            stats['separation'] = {
                'd_prime': float(d_prime),
                'auc_pphon_mean': float(auc_pphon_mean),
                'auc_pphon_min': float(auc_pphon_min),
            }
            
            # === NEW: Feature Comparison AUC (comprehensive) ===
            stats['feature_auc'] = {
                'P_utt': float(auc_putt),
                'P_phon_mean': float(auc_pphon_mean),
                'P_phon_min': float(auc_pphon_min),
                'P_utt_x_P_phon_min': float(auc_combined),
            }
        
        # === 3. Easy vs Hard（如果有標籤）===
        difficulty = self.data['difficulty']
        easy_mask = difficulty == 'easy'
        hard_mask = difficulty == 'hard'
        
        if easy_mask.any() or hard_mask.any():
            # 只看負樣本（應該拒絕的）
            easy_neg = easy_mask & neg_mask
            hard_neg = hard_mask & neg_mask
            
            stats['easy_vs_hard'] = {
                'easy_negative_pphon_mean': float(self.data['pphon_mean'][easy_neg].mean()) if easy_neg.any() else None,
                'hard_negative_pphon_mean': float(self.data['pphon_mean'][hard_neg].mean()) if hard_neg.any() else None,
                'easy_negative_pphon_min': float(self.data['pphon_min'][easy_neg].mean()) if easy_neg.any() else None,
                'hard_negative_pphon_min': float(self.data['pphon_min'][hard_neg].mean()) if hard_neg.any() else None,
                'hard_neg_count': int(hard_neg.sum()),
                'easy_neg_count': int(easy_neg.sum()),
            }
            
            # === NEW: Separate AUC for Easy and Hard subsets ===
            easy_pos = easy_mask & pos_mask
            hard_pos = hard_mask & pos_mask
            
            # Easy subset AUC (all features)
            if easy_pos.any() and easy_neg.any():
                easy_labels = np.concatenate([np.ones(easy_pos.sum()), np.zeros(easy_neg.sum())])
                
                easy_putt = np.concatenate([self.data['putt'][easy_pos], self.data['putt'][easy_neg]])
                easy_pphon_min = np.concatenate([self.data['pphon_min'][easy_pos], self.data['pphon_min'][easy_neg]])
                easy_combined = easy_putt * easy_pphon_min
                
                stats['easy_vs_hard']['easy_auc_P_utt'] = float(roc_auc_score(easy_labels, easy_putt))
                stats['easy_vs_hard']['easy_auc_P_phon_min'] = float(roc_auc_score(easy_labels, easy_pphon_min))
                stats['easy_vs_hard']['easy_auc_combined'] = float(roc_auc_score(easy_labels, easy_combined))
            
            # Hard subset AUC (all features)
            if hard_pos.any() and hard_neg.any():
                hard_labels = np.concatenate([np.ones(hard_pos.sum()), np.zeros(hard_neg.sum())])
                
                hard_putt = np.concatenate([self.data['putt'][hard_pos], self.data['putt'][hard_neg]])
                hard_pphon_min = np.concatenate([self.data['pphon_min'][hard_pos], self.data['pphon_min'][hard_neg]])
                hard_combined = hard_putt * hard_pphon_min
                
                stats['easy_vs_hard']['hard_auc_P_utt'] = float(roc_auc_score(hard_labels, hard_putt))
                stats['easy_vs_hard']['hard_auc_P_phon_min'] = float(roc_auc_score(hard_labels, hard_pphon_min))
                stats['easy_vs_hard']['hard_auc_combined'] = float(roc_auc_score(hard_labels, hard_combined))
        
        # === 4. TP vs FP 分析 ===
        P_utt = self.data['putt']
        threshold = 0.5
        
        predicted_pos = P_utt > threshold
        predicted_neg = P_utt <= threshold
        
        TP_mask = predicted_pos & pos_mask
        FP_mask = predicted_pos & neg_mask
        TN_mask = predicted_neg & neg_mask
        FN_mask = predicted_neg & pos_mask
        
        stats['confusion'] = {
            'TP_count': int(TP_mask.sum()),
            'FP_count': int(FP_mask.sum()),
            'TN_count': int(TN_mask.sum()),
            'FN_count': int(FN_mask.sum()),
            
            'TP_pphon_mean': float(self.data['pphon_mean'][TP_mask].mean()) if TP_mask.any() else None,
            'TP_pphon_min': float(self.data['pphon_min'][TP_mask].mean()) if TP_mask.any() else None,
            
            'FP_pphon_mean': float(self.data['pphon_mean'][FP_mask].mean()) if FP_mask.any() else None,
            'FP_pphon_min': float(self.data['pphon_min'][FP_mask].mean()) if FP_mask.any() else None,
            
            'FN_pphon_mean': float(self.data['pphon_mean'][FN_mask].mean()) if FN_mask.any() else None,
            'FN_pphon_min': float(self.data['pphon_min'][FN_mask].mean()) if FN_mask.any() else None,
        }
        
        # 關鍵指標：FP 的 pphon_min 是否顯著低於 TP？
        if TP_mask.any() and FP_mask.any():
            stats['confusion']['TP_FP_pphon_min_gap'] = float(
                self.data['pphon_min'][TP_mask].mean() - self.data['pphon_min'][FP_mask].mean()
            )
        
        # === NEW: High-Confidence False Positive Analysis ===
        high_conf_threshold = 0.8
        high_conf_FP_mask = (P_utt > high_conf_threshold) & neg_mask
        
        if high_conf_FP_mask.any():
            hc_fp_pphon_min = self.data['pphon_min'][high_conf_FP_mask]
            hc_fp_pphon_mean = self.data['pphon_mean'][high_conf_FP_mask]
            
            stats['high_confidence_fp'] = {
                'count': int(high_conf_FP_mask.sum()),
                'percentage_of_all_fp': float(high_conf_FP_mask.sum() / FP_mask.sum() * 100) if FP_mask.any() else 0,
                'pphon_min_mean': float(hc_fp_pphon_min.mean()),
                'pphon_min_std': float(hc_fp_pphon_min.std()),
                'pphon_mean_mean': float(hc_fp_pphon_mean.mean()),
                # Key insight: how many can be filtered by P_phon_min < 0.5?
                'filterable_by_pphon_min_lt_0.5': float((hc_fp_pphon_min < 0.5).mean() * 100),
                'filterable_by_pphon_min_lt_0.3': float((hc_fp_pphon_min < 0.3).mean() * 100),
            }
            
            # Breakdown by difficulty
            hc_fp_easy = high_conf_FP_mask & easy_mask
            hc_fp_hard = high_conf_FP_mask & hard_mask
            if hc_fp_easy.any():
                stats['high_confidence_fp']['easy_count'] = int(hc_fp_easy.sum())
                stats['high_confidence_fp']['easy_pphon_min_mean'] = float(self.data['pphon_min'][hc_fp_easy].mean())
            if hc_fp_hard.any():
                stats['high_confidence_fp']['hard_count'] = int(hc_fp_hard.sum())
                stats['high_confidence_fp']['hard_pphon_min_mean'] = float(self.data['pphon_min'][hc_fp_hard].mean())
        
        # === NEW: Hard Negatives Detailed Analysis ===
        hard_neg_mask = hard_mask & neg_mask
        if hard_neg_mask.any():
            hard_neg_high_putt = hard_neg_mask & (P_utt > 0.5)
            hard_neg_very_high_putt = hard_neg_mask & (P_utt > 0.8)
            
            stats['hard_analysis'] = {
                'hard_neg_total': int(hard_neg_mask.sum()),
                'hard_neg_high_putt_count': int(hard_neg_high_putt.sum()),
                'hard_neg_high_putt_ratio': float(hard_neg_high_putt.sum() / hard_neg_mask.sum()) if hard_neg_mask.any() else 0,
                'hard_neg_very_high_putt_count': int(hard_neg_very_high_putt.sum()),
                'hard_neg_very_high_putt_ratio': float(hard_neg_very_high_putt.sum() / hard_neg_mask.sum()) if hard_neg_mask.any() else 0,
            }
            
            if hard_neg_high_putt.any():
                stats['hard_analysis']['hard_neg_high_putt_pphon_mean'] = float(self.data['pphon_mean'][hard_neg_high_putt].mean())
                stats['hard_analysis']['hard_neg_high_putt_pphon_min'] = float(self.data['pphon_min'][hard_neg_high_putt].mean())
                stats['hard_analysis']['hard_neg_high_putt_pphon_min_std'] = float(self.data['pphon_min'][hard_neg_high_putt].std())
                # How many can be filtered by P_phon_min?
                filterable = (self.data['pphon_min'][hard_neg_high_putt] < 0.5).sum()
                stats['hard_analysis']['hard_neg_high_putt_filterable_by_pphon_lt_0.5'] = float(filterable / hard_neg_high_putt.sum() * 100)
        
        # === 5. 按 Category 分析 ===
        categories = self.data['category']
        unique_cats = np.unique(categories)
        
        stats['by_category'] = {}
        for cat in unique_cats:
            if cat == 'unknown':
                continue
            cat_mask = categories == cat
        
        # === 6. P_spk 分析（Personalized KWS 專用）===
        z_spk = self.data['z_speaker']
        if z_spk.any():  # Only if speaker labels are available
            same_spk = z_spk == 1
            diff_spk = z_spk == 0
            
            stats['by_speaker'] = {
                'same_speaker_count': int(same_spk.sum()),
                'diff_speaker_count': int(diff_spk.sum()),
                
                # P_spk for same vs different speakers
                'same_speaker_pspk_mean': float(self.data['pspk'][same_spk].mean()) if same_spk.any() else None,
                'diff_speaker_pspk_mean': float(self.data['pspk'][diff_spk].mean()) if diff_spk.any() else None,
                
                # P_phon for same vs different speakers (to see if personalization affects P_phon)
                'same_speaker_pphon_mean': float(self.data['pphon_mean'][same_spk].mean()) if same_spk.any() else None,
                'diff_speaker_pphon_mean': float(self.data['pphon_mean'][diff_spk].mean()) if diff_spk.any() else None,
            }
            
            # P_spk separation (how well does P_spk distinguish speakers?)
            if same_spk.any() and diff_spk.any():
                pspk_same = self.data['pspk'][same_spk]
                pspk_diff = self.data['pspk'][diff_spk]
                
                # AUC for P_spk as speaker classifier
                all_pspk = np.concatenate([pspk_same, pspk_diff])
                all_spk_labels = np.concatenate([np.ones(len(pspk_same)), np.zeros(len(pspk_diff))])
                auc_pspk = roc_auc_score(all_spk_labels, all_pspk)
                
                stats['by_speaker']['auc_pspk'] = float(auc_pspk)
        
        # === 7. Fusion Score 分析 (P_utt × P_spk) ===
        if len(self.data['score']) > 0:
            stats['fusion_score'] = {
                'score_mean': float(self.data['score'].mean()),
                'score_std': float(self.data['score'].std()),
                
                # Score for positive (ts-tk) vs negative samples
                'positive_score_mean': float(self.data['score'][pos_mask].mean()) if pos_mask.any() else None,
                'negative_score_mean': float(self.data['score'][neg_mask].mean()) if neg_mask.any() else None,
            }
            
            # AUC using fusion score (for final TO-KWS performance)
            if pos_mask.any() and neg_mask.any():
                pos_scores = self.data['score'][pos_mask]
                neg_scores = self.data['score'][neg_mask]
                all_fusion_scores = np.concatenate([pos_scores, neg_scores])
                all_labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
                auc_fusion = roc_auc_score(all_labels, all_fusion_scores)
                stats['fusion_score']['auc_fusion'] = float(auc_fusion)
                
                # Compare: does fusion score improve over P_utt alone?
                auc_putt_only = roc_auc_score(all_labels, np.concatenate([
                    self.data['putt'][pos_mask], self.data['putt'][neg_mask]
                ]))
                stats['fusion_score']['auc_putt_only'] = float(auc_putt_only)
                stats['fusion_score']['fusion_improvement'] = float(auc_fusion - auc_putt_only)
            if cat_mask.any():
                stats['by_category'][cat] = {
                    'count': int(cat_mask.sum()),
                    'pphon_mean': float(self.data['pphon_mean'][cat_mask].mean()),
                    'pphon_min': float(self.data['pphon_min'][cat_mask].mean()),
                }
        
        return stats
    
    def collect_error_cases(self, max_cases=50):
        """收集錯誤案例詳情"""
        P_utt = self.data['putt']
        z_kw = self.data['z_keyword']
        threshold = 0.5
        
        predicted_pos = P_utt > threshold
        FP_mask = predicted_pos & (z_kw == 0)
        FN_mask = (~predicted_pos) & (z_kw == 1)
        
        error_cases = []
        
        # False Positives - sort by P_utt descending (most confident errors first)
        fp_indices = np.where(FP_mask)[0]
        fp_indices = fp_indices[np.argsort(P_utt[fp_indices])[::-1]][:max_cases]
        for idx in fp_indices:
            case = {
                'type': 'FP',
                'index': int(idx),
                'P_utt': float(P_utt[idx]),
                'P_phon_mean': float(self.data['pphon_mean'][idx]),
                'P_phon_min': float(self.data['pphon_min'][idx]),
                'P_phon_max': float(self.data['pphon_max'][idx]),
                'difficulty': self.data['difficulty'][idx],
                'category': self.data['category'][idx],
                # NEW: Add audio_path and keyword_text for manual investigation
                'audio_path': self.data['audio_path'][idx] if len(self.data['audio_path']) > idx else '',
                'keyword_text': self.data['keyword_text'][idx] if len(self.data['keyword_text']) > idx else '',
            }
            error_cases.append(case)
        
        # False Negatives - sort by P_utt ascending (most confident misses first)
        fn_indices = np.where(FN_mask)[0]
        fn_indices = fn_indices[np.argsort(P_utt[fn_indices])][:max_cases]
        for idx in fn_indices:
            case = {
                'type': 'FN',
                'index': int(idx),
                'P_utt': float(P_utt[idx]),
                'P_phon_mean': float(self.data['pphon_mean'][idx]),
                'P_phon_min': float(self.data['pphon_min'][idx]),
                'P_phon_max': float(self.data['pphon_max'][idx]),
                'difficulty': self.data['difficulty'][idx],
                'category': self.data['category'][idx],
                # NEW: Add audio_path and keyword_text for manual investigation
                'audio_path': self.data['audio_path'][idx] if len(self.data['audio_path']) > idx else '',
                'keyword_text': self.data['keyword_text'][idx] if len(self.data['keyword_text']) > idx else '',
            }
            error_cases.append(case)
        
        return error_cases
    
    def _plot_distribution_subset(self, subset_mask, suffix, title_suffix):
        """Helper: Plot distribution for a specific subset (all, easy, or hard)"""
        z_kw = self.data['z_keyword']
        pos_mask = (z_kw == 1) & subset_mask
        neg_mask = (z_kw == 0) & subset_mask
        
        if not (pos_mask.any() and neg_mask.any()):
            return False
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # P_phon mean 分布
        ax = axes[0, 0]
        ax.hist(self.data['pphon_mean'][pos_mask], bins=30, alpha=0.6, 
                label=f'Positive (n={pos_mask.sum()})', color='green')
        ax.hist(self.data['pphon_mean'][neg_mask], bins=30, alpha=0.6, 
                label=f'Negative (n={neg_mask.sum()})', color='red')
        ax.set_xlabel('P_phon Mean')
        ax.set_ylabel('Count')
        ax.set_title('P_phon Mean Distribution')
        ax.legend()
        
        # P_phon min 分布
        ax = axes[0, 1]
        ax.hist(self.data['pphon_min'][pos_mask], bins=30, alpha=0.6, 
                label='Positive', color='green')
        ax.hist(self.data['pphon_min'][neg_mask], bins=30, alpha=0.6, 
                label='Negative', color='red')
        ax.set_xlabel('P_phon Min')
        ax.set_ylabel('Count')
        ax.set_title('P_phon Min Distribution')
        ax.legend()
        
        # P_utt vs P_phon_mean scatter
        ax = axes[1, 0]
        ax.scatter(self.data['putt'][pos_mask], self.data['pphon_mean'][pos_mask], 
                   alpha=0.3, label='Positive', color='green', s=10)
        ax.scatter(self.data['putt'][neg_mask], self.data['pphon_mean'][neg_mask], 
                   alpha=0.3, label='Negative', color='red', s=10)
        ax.set_xlabel('P_utt')
        ax.set_ylabel('P_phon Mean')
        ax.set_title('P_utt vs P_phon Mean')
        ax.legend()
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # P_utt vs P_phon_min scatter
        ax = axes[1, 1]
        ax.scatter(self.data['putt'][pos_mask], self.data['pphon_min'][pos_mask], 
                   alpha=0.3, label='Positive', color='green', s=10)
        ax.scatter(self.data['putt'][neg_mask], self.data['pphon_min'][neg_mask], 
                   alpha=0.3, label='Negative', color='red', s=10)
        ax.set_xlabel('P_utt')
        ax.set_ylabel('P_phon Min')
        ax.set_title('P_utt vs P_phon Min')
        ax.legend()
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        
        plt.suptitle(f'P_phon Distribution {title_suffix}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'pphon_distribution{suffix}.png'), dpi=150)
        plt.close()
        return True
    
    def plot_distribution(self):
        """繪製 P_phon 分布直方圖 - 分別產生 all/easy/hard 三張圖"""
        difficulty = self.data['difficulty']
        easy_mask = difficulty == 'easy'
        hard_mask = difficulty == 'hard'
        all_mask = np.ones(len(difficulty), dtype=bool)
        
        # All samples
        if self._plot_distribution_subset(all_mask, '', '(All Samples)'):
            print(f"Saved: pphon_distribution.png")
        
        # Easy only
        if easy_mask.any() and self._plot_distribution_subset(easy_mask, '_easy', '(Easy Only)'):
            print(f"Saved: pphon_distribution_easy.png")
        
        # Hard only
        if hard_mask.any() and self._plot_distribution_subset(hard_mask, '_hard', '(Hard Only)'):
            print(f"Saved: pphon_distribution_hard.png")
    
    def _plot_tp_vs_fp_subset(self, subset_mask, suffix, title_suffix):
        """Helper: Plot TP vs FP for a specific subset"""
        P_utt = self.data['putt']
        z_kw = self.data['z_keyword']
        threshold = 0.5
        
        predicted_pos = P_utt > threshold
        TP_mask = predicted_pos & (z_kw == 1) & subset_mask
        FP_mask = predicted_pos & (z_kw == 0) & subset_mask
        
        if not (TP_mask.any() and FP_mask.any()):
            return False
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # P_phon mean
        ax = axes[0]
        data_to_plot = [
            self.data['pphon_mean'][TP_mask],
            self.data['pphon_mean'][FP_mask]
        ]
        ax.boxplot(data_to_plot, tick_labels=['True Positive', 'False Positive'])
        ax.set_ylabel('P_phon Mean')
        ax.set_title('P_phon Mean: TP vs FP')
        
        for i, d in enumerate(data_to_plot):
            ax.annotate(f'μ={d.mean():.3f}', xy=(i+1, d.mean()), 
                       xytext=(i+1.2, d.mean()), fontsize=10)
        
        # P_phon min
        ax = axes[1]
        data_to_plot = [
            self.data['pphon_min'][TP_mask],
            self.data['pphon_min'][FP_mask]
        ]
        ax.boxplot(data_to_plot, tick_labels=['True Positive', 'False Positive'])
        ax.set_ylabel('P_phon Min')
        ax.set_title('P_phon Min: TP vs FP\n(Gap > 0.1 suggests P_phon can help filter FP)')
        
        for i, d in enumerate(data_to_plot):
            ax.annotate(f'μ={d.mean():.3f}', xy=(i+1, d.mean()), 
                       xytext=(i+1.2, d.mean()), fontsize=10)
        
        plt.suptitle(f'TP vs FP Analysis {title_suffix}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'pphon_tp_vs_fp{suffix}.png'), dpi=150)
        plt.close()
        return True
    
    def plot_tp_vs_fp(self):
        """繪製 TP vs FP 的 P_phon 對比 - 分別產生 all/easy/hard 三張圖"""
        difficulty = self.data['difficulty']
        easy_mask = difficulty == 'easy'
        hard_mask = difficulty == 'hard'
        all_mask = np.ones(len(difficulty), dtype=bool)
        
        # All samples
        if self._plot_tp_vs_fp_subset(all_mask, '', '(All Samples)'):
            print(f"Saved: pphon_tp_vs_fp.png")
        
        # Easy only
        if easy_mask.any() and self._plot_tp_vs_fp_subset(easy_mask, '_easy', '(Easy Only)'):
            print(f"Saved: pphon_tp_vs_fp_easy.png")
        
        # Hard only
        if hard_mask.any() and self._plot_tp_vs_fp_subset(hard_mask, '_hard', '(Hard Only)'):
            print(f"Saved: pphon_tp_vs_fp_hard.png")
    
    def plot_easy_vs_hard(self):
        """繪製 Easy vs Hard 對比（如果有標籤）"""
        difficulty = self.data['difficulty']
        z_kw = self.data['z_keyword']
        
        easy_mask = difficulty == 'easy'
        hard_mask = difficulty == 'hard'
        neg_mask = z_kw == 0
        
        easy_neg = easy_mask & neg_mask
        hard_neg = hard_mask & neg_mask
        
        if not (easy_neg.any() and hard_neg.any()):
            print("Not enough Easy/Hard negative samples")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # P_phon mean for negatives
        ax = axes[0]
        data_to_plot = [
            self.data['pphon_mean'][easy_neg],
            self.data['pphon_mean'][hard_neg]
        ]
        ax.boxplot(data_to_plot, labels=['Easy Negative', 'Hard Negative'])
        ax.set_ylabel('P_phon Mean')
        ax.set_title('Negative Samples: Easy vs Hard\n(Lower is better for negatives)')
        
        for i, d in enumerate(data_to_plot):
            ax.annotate(f'μ={d.mean():.3f}', xy=(i+1, d.mean()), 
                       xytext=(i+1.2, d.mean()), fontsize=10)
        
        # P_phon min for negatives
        ax = axes[1]
        data_to_plot = [
            self.data['pphon_min'][easy_neg],
            self.data['pphon_min'][hard_neg]
        ]
        ax.boxplot(data_to_plot, labels=['Easy Negative', 'Hard Negative'])
        ax.set_ylabel('P_phon Min')
        ax.set_title('Negative Samples P_phon Min: Easy vs Hard')
        
        for i, d in enumerate(data_to_plot):
            ax.annotate(f'μ={d.mean():.3f}', xy=(i+1, d.mean()), 
                       xytext=(i+1.2, d.mean()), fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'easy_vs_hard.png'), dpi=150)
        plt.close()
        print(f"Saved: easy_vs_hard.png")
    
    def plot_roc_curves(self):
        """繪製不同特徵的 ROC 曲線比較"""
        from sklearn.metrics import roc_curve, auc
        
        z_kw = self.data['z_keyword']
        difficulty = self.data['difficulty']
        easy_mask = difficulty == 'easy'
        hard_mask = difficulty == 'hard'
        pos_mask = z_kw == 1
        neg_mask = z_kw == 0
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for ax_idx, (subset_mask, subset_name) in enumerate([(easy_mask, 'Easy'), (hard_mask, 'Hard')]):
            ax = axes[ax_idx]
            
            subset_pos = subset_mask & pos_mask
            subset_neg = subset_mask & neg_mask
            
            if not (subset_pos.any() and subset_neg.any()):
                ax.text(0.5, 0.5, f'No data for {subset_name}', ha='center', va='center')
                ax.set_title(f'{subset_name} Subset')
                continue
            
            # Prepare features
            labels = np.concatenate([np.ones(subset_pos.sum()), np.zeros(subset_neg.sum())])
            
            features = {
                'P_utt': np.concatenate([self.data['putt'][subset_pos], self.data['putt'][subset_neg]]),
                'P_phon_min': np.concatenate([self.data['pphon_min'][subset_pos], self.data['pphon_min'][subset_neg]]),
            }
            features['P_utt × P_phon_min'] = features['P_utt'] * features['P_phon_min']
            
            colors = {'P_utt': '#3498db', 'P_phon_min': '#e74c3c', 'P_utt × P_phon_min': '#2ecc71'}
            
            for feat_name, feat_values in features.items():
                fpr, tpr, _ = roc_curve(labels, feat_values)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=colors[feat_name], lw=2, 
                       label=f'{feat_name} (AUC = {roc_auc:.4f})')
            
            ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.5)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curves ({subset_name} Subset)')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'roc_curves.png'), dpi=150)
        plt.close()
        print(f"Saved: roc_curves.png")
    
    def print_summary(self, stats):
        """印出分析摘要"""
        print("\n" + "="*70)
        print("P_phon Analysis Summary")
        print("="*70)
        
        if 'error' in stats:
            print(f"Error: {stats['error']}")
            return
        
        basic = stats['basic']
        
        print(f"\n[Basic Statistics]")
        print(f"  Total samples: {basic['total_samples']} "
              f"(pos: {basic['positive_samples']}, neg: {basic['negative_samples']})")
        print(f"  Positive P_phon mean: {basic['positive_pphon_mean']:.4f} ± {basic['positive_pphon_mean_std']:.4f}")
        print(f"  Negative P_phon mean: {basic['negative_pphon_mean']:.4f} ± {basic['negative_pphon_mean_std']:.4f}")
        print(f"  Gap (mean): {basic['gap_mean']:.4f}")
        print(f"  Gap (min):  {basic['gap_min']:.4f}")
        
        if 'separation' in stats:
            sep = stats['separation']
            print(f"\n[Separation Metrics]")
            print(f"  d-prime:        {sep['d_prime']:.4f}")
            print(f"  AUC (pphon_mean): {sep['auc_pphon_mean']:.4f}")
            print(f"  AUC (pphon_min):  {sep['auc_pphon_min']:.4f}")
        
        # === NEW: Feature AUC Comparison ===
        if 'feature_auc' in stats:
            fa = stats['feature_auc']
            print(f"\n[Feature AUC Comparison]")
            print(f"  {'Feature':<20} {'AUC':>8}")
            print(f"  {'-'*30}")
            for feat, auc in sorted(fa.items(), key=lambda x: x[1], reverse=True):
                marker = "★" if auc == max(fa.values()) else " "
                print(f"  {feat:<20} {auc:>7.4f} {marker}")
        
        if 'confusion' in stats:
            conf = stats['confusion']
            print(f"\n[TP vs FP Analysis]")
            print(f"  TP count: {conf['TP_count']}, FP count: {conf['FP_count']}")
            if conf.get('TP_pphon_min') is not None and conf.get('FP_pphon_min') is not None:
                print(f"  TP P_phon min: {conf['TP_pphon_min']:.4f}")
                print(f"  FP P_phon min: {conf['FP_pphon_min']:.4f}")
                if 'TP_FP_pphon_min_gap' in conf:
                    gap = conf['TP_FP_pphon_min_gap']
                    print(f"  Gap: {gap:.4f}")
                    if gap > 0.1:
                        print(f"  → ✓ P_phon_min CAN help filter false positives!")
                    else:
                        print(f"  → △ P_phon_min has LIMITED additional value")
        
        # === NEW: High-Confidence FP Analysis ===
        if 'high_confidence_fp' in stats:
            hc = stats['high_confidence_fp']
            print(f"\n[High-Confidence False Positives (P_utt > 0.8)]")
            print(f"  Count: {hc['count']} ({hc['percentage_of_all_fp']:.1f}% of all FP)")
            print(f"  P_phon_min mean: {hc['pphon_min_mean']:.4f} ± {hc['pphon_min_std']:.4f}")
            print(f"  Filterable by P_phon_min < 0.5: {hc['filterable_by_pphon_min_lt_0.5']:.1f}%")
            print(f"  Filterable by P_phon_min < 0.3: {hc['filterable_by_pphon_min_lt_0.3']:.1f}%")
            if hc.get('easy_count') is not None:
                print(f"  Easy: {hc['easy_count']} (P_phon_min mean: {hc.get('easy_pphon_min_mean', 0):.4f})")
            if hc.get('hard_count') is not None:
                print(f"  Hard: {hc['hard_count']} (P_phon_min mean: {hc.get('hard_pphon_min_mean', 0):.4f})")
        
        if 'easy_vs_hard' in stats:
            eh = stats['easy_vs_hard']
            print(f"\n[Easy vs Hard Analysis]")
            if eh.get('easy_neg_count') is not None:
                print(f"  Easy samples: {eh['easy_neg_count']} neg")
            if eh.get('hard_neg_count') is not None:
                print(f"  Hard samples: {eh['hard_neg_count']} neg")
            
            if eh.get('easy_negative_pphon_mean') is not None and eh.get('hard_negative_pphon_mean') is not None:
                print(f"\n  P_phon (negatives only):")
                print(f"    Easy: mean={eh['easy_negative_pphon_mean']:.4f}, min={eh.get('easy_negative_pphon_min', 0):.4f}")
                print(f"    Hard: mean={eh['hard_negative_pphon_mean']:.4f}, min={eh.get('hard_negative_pphon_min', 0):.4f}")
                
                if eh['hard_negative_pphon_mean'] > eh['easy_negative_pphon_mean'] + 0.05:
                    print(f"  → Hard negatives have higher P_phon (harder to reject)")
            
            # NEW: Easy/Hard separate AUCs
            if eh.get('easy_auc_P_utt') is not None:
                print(f"\n  Easy AUC: P_utt={eh['easy_auc_P_utt']:.4f}, P_phon_min={eh['easy_auc_P_phon_min']:.4f}, Combined={eh['easy_auc_combined']:.4f}")
            if eh.get('hard_auc_P_utt') is not None:
                print(f"  Hard AUC: P_utt={eh['hard_auc_P_utt']:.4f}, P_phon_min={eh['hard_auc_P_phon_min']:.4f}, Combined={eh['hard_auc_combined']:.4f}")
        
        if 'by_category' in stats and stats['by_category']:
            print(f"\n[By Category]")
            for cat, cat_stats in stats['by_category'].items():
                print(f"  {cat}: n={cat_stats['count']}, "
                      f"pphon_mean={cat_stats['pphon_mean']:.4f}, "
                      f"pphon_min={cat_stats['pphon_min']:.4f}")
        
        # === NEW: Hard Negatives Analysis ===
        if 'hard_analysis' in stats:
            ha = stats['hard_analysis']
            print(f"\n[Hard Negatives Analysis]")
            print(f"  Total Hard Negatives: {ha.get('hard_neg_total', 0)}")
            print(f"  With P_utt > 0.5: {ha.get('hard_neg_high_putt_count', 0)} ({ha.get('hard_neg_high_putt_ratio', 0)*100:.1f}%)")
            print(f"  With P_utt > 0.8: {ha.get('hard_neg_very_high_putt_count', 0)} ({ha.get('hard_neg_very_high_putt_ratio', 0)*100:.1f}%)")
            if ha.get('hard_neg_high_putt_pphon_min') is not None:
                print(f"  High-P_utt Hard Neg P_phon_min: {ha['hard_neg_high_putt_pphon_min']:.4f} ± {ha.get('hard_neg_high_putt_pphon_min_std', 0):.4f}")
                print(f"  Filterable by P_phon_min < 0.5: {ha.get('hard_neg_high_putt_filterable_by_pphon_lt_0.5', 0):.1f}%")
        
        # === NEW: Improvement Suggestions ===
        print(f"\n[💡 Improvement Suggestions]")
        suggestions = []
        
        # Check Hard AUC
        if 'easy_vs_hard' in stats:
            eh = stats['easy_vs_hard']
            if eh.get('hard_auc_P_utt') is not None and eh['hard_auc_P_utt'] < 0.90:
                suggestions.append(f"⚠️ Hard AUC ({eh['hard_auc_P_utt']:.4f}) < 0.90: Consider Hard Negative Mining or Focal Loss")
            if eh.get('hard_auc_combined') is not None and eh.get('hard_auc_P_utt') is not None:
                improvement = eh['hard_auc_combined'] - eh['hard_auc_P_utt']
                if improvement > 0.01:
                    suggestions.append(f"✓ P_phon integration improves Hard AUC by {improvement:.4f}")
        
        # Check High-confidence FP distribution
        if 'high_confidence_fp' in stats:
            hc = stats['high_confidence_fp']
            easy_count = hc.get('easy_count', 0)
            hard_count = hc.get('hard_count', 0)
            if hard_count > easy_count * 5:
                suggestions.append(f"⚠️ High-conf FP ratio Hard/Easy = {hard_count/max(easy_count,1):.1f}x: Focus on phonetically similar pairs")
            
            filterable = hc.get('filterable_by_pphon_min_lt_0.5', 0)
            if filterable > 50:
                suggestions.append(f"✓ {filterable:.0f}% of high-conf FP can be filtered by P_phon_min < 0.5")
            elif filterable < 30:
                suggestions.append(f"⚠️ Only {filterable:.0f}% high-conf FP filterable by P_phon: Consider improving phoneme-level training")
        
        # Check Hard analysis
        if 'hard_analysis' in stats:
            ha = stats['hard_analysis']
            if ha.get('hard_neg_high_putt_ratio', 0) > 0.3:
                suggestions.append(f"⚠️ {ha['hard_neg_high_putt_ratio']*100:.1f}% of Hard Neg have P_utt > 0.5: Need better hard negative handling")
        
        # Check TP/FP gap
        if 'confusion' in stats:
            gap = stats['confusion'].get('TP_FP_pphon_min_gap', 0)
            if gap < 0.1:
                suggestions.append(f"⚠️ TP-FP P_phon_min gap ({gap:.4f}) < 0.1: P_phon_min has limited distinguishing power")
        
        if suggestions:
            for s in suggestions:
                print(f"  {s}")
        else:
            print(f"  ✓ No obvious issues detected. Consider further experiments.")
        
        print("="*70)
    
    def save_results(self):
        """儲存所有結果"""
        # 計算統計
        stats = self.compute_statistics()
        
        # 收集錯誤案例
        error_cases = self.collect_error_cases()
        
        # 儲存 JSON
        results = {
            'statistics': stats,
            'error_case_count': len(error_cases),
        }
        
        with open(os.path.join(self.output_dir, 'pphon_stats.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved: pphon_stats.json")
        
        # 儲存錯誤案例 CSV
        if error_cases:
            df = pd.DataFrame(error_cases)
            df.to_csv(os.path.join(self.output_dir, 'error_cases.csv'), index=False)
            print(f"Saved: error_cases.csv")
        
        # 繪製圖表
        self.plot_distribution()
        self.plot_tp_vs_fp()
        self.plot_easy_vs_hard()
        self.plot_roc_curves()  # NEW: ROC curves
        
        # NEW: 繪製 t-SNE 視覺化（如果有 embedding）
        if self.collect_embeddings:
            self.plot_embedding_tsne()
        
        # 印出摘要並記錄到檔案
        import io
        import sys
        
        # Capture print output
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        self.print_summary(stats)
        
        # Restore stdout and get captured text
        sys.stdout = old_stdout
        summary_text = captured_output.getvalue()
        
        # Print to console
        print(summary_text)
        
        # Save to log file
        log_path = os.path.join(self.output_dir, 'analysis_log.txt')
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        print(f"Saved: analysis_log.txt")
        
        return stats
    
    def plot_embedding_tsne(self, max_samples=5000, perplexity=30, random_state=42):
        """
        繪製 embedding 的 t-SNE 2D 視覺化
        
        Args:
            max_samples: 最多使用多少樣本（t-SNE 對大量樣本很慢）
            perplexity: t-SNE perplexity 參數
            random_state: 隨機種子
        """
        print("\n🎨 Generating t-SNE visualizations...")
        
        # 檢查是否有 embedding
        if not self.collect_embeddings:
            print("⚠️ No embeddings collected. Call model with return_embeddings=True.")
            return
        
        # 準備 labels
        z_kw = self.data['z_keyword']
        z_spk = self.data['z_speaker']
        categories = self.data['category']
        n_samples = len(z_kw)
        
        # 下采樣（t-SNE 對大量樣本很慢）
        if n_samples > max_samples:
            np.random.seed(random_state)
            indices = np.random.choice(n_samples, max_samples, replace=False)
            print(f"  Subsampling: {n_samples} → {max_samples} samples")
        else:
            indices = np.arange(n_samples)
        
        # 4-class labeling: keyword (pos/neg) x speaker (same/diff)
        labels_4class = []
        for i in indices:
            if z_kw[i] == 1 and z_spk[i] == 1:
                labels_4class.append('ts-tk')
            elif z_kw[i] == 1 and z_spk[i] == 0:
                labels_4class.append('nts-tk')
            elif z_kw[i] == 0 and z_spk[i] == 1:
                labels_4class.append('ts-ntk')
            else:
                labels_4class.append('nts-ntk')
        labels_4class = np.array(labels_4class)
        
        # Color mapping
        color_map = {
            'ts-tk': '#2ecc71',    # Green: target speaker, target keyword (should accept)
            'nts-tk': '#e74c3c',   # Red: non-target speaker, target keyword
            'ts-ntk': '#3498db',   # Blue: target speaker, non-target keyword
            'nts-ntk': '#95a5a6',  # Gray: non-target speaker, non-target keyword
        }
        
        # 繪製每種 embedding 的 t-SNE
        embedding_types = [
            ('emb_audio', 'Audio Embeddings (after FiLM)'),
            ('emb_text', 'Text Embeddings'),
            ('emb_attention', 'Attention Output Embeddings'),
        ]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for ax_idx, (emb_key, title) in enumerate(embedding_types):
            ax = axes[ax_idx]
            
            if len(self.embeddings[emb_key]) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(title)
                continue
            
            # 取出子集
            emb_subset = self.embeddings[emb_key][indices]
            
            # t-SNE 降維
            print(f"  Computing t-SNE for {emb_key}...")
            tsne = TSNE(
                n_components=2, 
                perplexity=min(perplexity, len(indices) - 1),
                random_state=random_state,
                init='pca',
                learning_rate='auto'
            )
            emb_2d = tsne.fit_transform(emb_subset)
            
            # 繪製散點圖
            for cat in ['nts-ntk', 'ts-ntk', 'nts-tk', 'ts-tk']:  # 順序確保重要類別在上層
                mask = labels_4class == cat
                if mask.any():
                    ax.scatter(
                        emb_2d[mask, 0], emb_2d[mask, 1],
                        c=color_map[cat], label=cat,
                        alpha=0.6, s=15, edgecolors='none'
                    )
            
            ax.set_title(title)
            ax.set_xlabel('t-SNE Dim 1')
            ax.set_ylabel('t-SNE Dim 2')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Embedding t-SNE Visualization\n(Green=ts-tk should cluster separately)', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'embedding_tsne.png'), dpi=150)
        plt.close()
        print(f"Saved: embedding_tsne.png")
        
        # === NEW: Generate separate Easy/Hard t-SNE plots ===
        difficulty_subset = self.data['difficulty'][indices]
        easy_submask = difficulty_subset == 'easy'
        hard_submask = difficulty_subset == 'hard'
        
        for subset_name, submask in [('easy', easy_submask), ('hard', hard_submask)]:
            if not submask.any():
                continue
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            for ax_idx, (emb_key, title) in enumerate(embedding_types):
                ax = axes[ax_idx]
                
                if len(self.embeddings[emb_key]) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                    ax.set_title(title)
                    continue
                
                emb_subset = self.embeddings[emb_key][indices]
                
                # Compute t-SNE (same as before)
                tsne = TSNE(
                    n_components=2, 
                    perplexity=min(perplexity, submask.sum() - 1),
                    random_state=random_state,
                    init='pca',
                    learning_rate='auto'
                )
                emb_2d = tsne.fit_transform(emb_subset[submask])
                subset_labels = labels_4class[submask]
                
                for cat in ['nts-ntk', 'ts-ntk', 'nts-tk', 'ts-tk']:
                    cat_mask = subset_labels == cat
                    if cat_mask.any():
                        ax.scatter(
                            emb_2d[cat_mask, 0], emb_2d[cat_mask, 1],
                            c=color_map[cat], label=cat,
                            alpha=0.6, s=15, edgecolors='none'
                        )
                
                ax.set_title(title)
                ax.set_xlabel('t-SNE Dim 1')
                ax.set_ylabel('t-SNE Dim 2')
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)
            
            plt.suptitle(f'Embedding t-SNE Visualization ({subset_name.upper()} Only)', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'embedding_tsne_{subset_name}.png'), dpi=150)
            plt.close()
            print(f"Saved: embedding_tsne_{subset_name}.png")
        
        # === 額外：繪製 Audio embedding by P_phon (連續值著色) ===
        if len(self.embeddings['emb_audio']) > 0:
            self._plot_audio_embedding_by_pphon(indices, perplexity, random_state)

    def _plot_audio_embedding_by_pphon(self, indices, perplexity=30, random_state=42):
        """
        繪製 Audio embedding 以 P_phon_mean 著色 - 分別產生 all/easy/hard
        """
        difficulty_subset = self.data['difficulty'][indices]
        
        for subset_name, submask in [('', np.ones(len(indices), dtype=bool)), 
                                      ('_easy', difficulty_subset == 'easy'), 
                                      ('_hard', difficulty_subset == 'hard')]:
            if not submask.any() or submask.sum() < 50:  # Skip if too few samples
                continue
            
            sub_indices = np.array(indices)[submask]
            emb_subset = self.embeddings['emb_audio'][sub_indices]
            pphon_subset = self.data['pphon_mean'][sub_indices]
            z_kw_subset = self.data['z_keyword'][sub_indices]
            
            # t-SNE
            tsne = TSNE(
                n_components=2,
                perplexity=min(perplexity, len(sub_indices) - 1),
                random_state=random_state,
                max_iter=1000,
                init='pca',
                learning_rate='auto'
            )
            emb_2d = tsne.fit_transform(emb_subset)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # 左圖：以 P_phon mean 著色
            ax = axes[0]
            scatter = ax.scatter(
                emb_2d[:, 0], emb_2d[:, 1],
                c=pphon_subset, cmap='RdYlGn', 
                alpha=0.6, s=15, edgecolors='none',
                vmin=0, vmax=1
            )
            plt.colorbar(scatter, ax=ax, label='P_phon Mean')
            ax.set_title('Audio Embedding colored by P_phon Mean')
            ax.set_xlabel('t-SNE Dim 1')
            ax.set_ylabel('t-SNE Dim 2')
            ax.grid(True, alpha=0.3)
            
            # 右圖：以 Keyword Label 著色
            ax = axes[1]
            colors = ['#e74c3c' if z == 0 else '#2ecc71' for z in z_kw_subset]
            ax.scatter(
                emb_2d[:, 0], emb_2d[:, 1],
                c=colors, alpha=0.6, s=15, edgecolors='none'
            )
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2ecc71', label='Positive (keyword match)'),
                Patch(facecolor='#e74c3c', label='Negative (no match)'),
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            ax.set_title('Audio Embedding colored by Keyword Label')
            ax.set_xlabel('t-SNE Dim 1')
            ax.set_ylabel('t-SNE Dim 2')
            ax.grid(True, alpha=0.3)
            
            title_suffix = '' if subset_name == '' else f' ({subset_name[1:].upper()} Only)'
            plt.suptitle(f'Audio Embedding Analysis{title_suffix}', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'audio_embedding_analysis{subset_name}.png'), dpi=150)
            plt.close()
            print(f"Saved: audio_embedding_analysis{subset_name}.png")


def load_model(checkpoint_path, args):
    """
    載入模型並返回配置
    
    Returns:
        model: P_UKWS model
        config: dict containing audio_input, text_input, etc.
    """
    # 讀取 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 嘗試從 checkpoint 讀取訓練配置 (FIX: prioritize train_args from checkpoint)
    train_args = checkpoint.get('args', None)
    
    # 模型配置 - prioritize train_args over command line args
    if train_args is not None:
        print("📋 Using config from checkpoint's train_args")
        config = {
            'vocab': getattr(train_args, 'vocab', 60),
            'text_input': getattr(train_args, 'text_input', 'g2p_embed'),
            'audio_input': getattr(train_args, 'audio_input', 'both'),
            'stack_extractor': getattr(train_args, 'stack_extractor', 1),
            'frame_length': getattr(train_args, 'frame_length', 512),
            'hop_length': getattr(train_args, 'hop_length', 160),
            'num_mel': 40,
            'sample_rate': getattr(train_args, 'sample_rate', 16000),
            'log_mel': getattr(train_args, 'log_mel', True),
            'mode': getattr(train_args, 'mode', 'TB-KWS'),
            'speaker_encoder_path': getattr(train_args, 'speaker_encoder_path', 
                                            'model/speaker/efficient_tdnn'),
            'freeze_speaker_encoder': True,
        }
        # Allow command line args to override specific settings if explicitly provided
        if hasattr(args, 'audio_input') and args.audio_input != 'both':
            config['audio_input'] = args.audio_input
            print(f"  ↳ Overriding audio_input with command line: {args.audio_input}")
            
        if hasattr(args, 'speaker_encoder_path') and args.speaker_encoder_path is not None:
            config['speaker_encoder_path'] = args.speaker_encoder_path
            print(f"  ↳ Overriding speaker_encoder_path with command line: {args.speaker_encoder_path}")
    else:
        print("⚠️ No train_args in checkpoint, using command line args (may cause mismatch!)")
        config = {
            'vocab': getattr(args, 'vocab', 60),
            'text_input': getattr(args, 'text_input', 'cmu'),
            'audio_input': getattr(args, 'audio_input', 'both'),
            'stack_extractor': getattr(args, 'stack_extractor', 1),
            'frame_length': getattr(args, 'frame_length', 512),
            'hop_length': getattr(args, 'hop_length', 160),
            'num_mel': 40,
            'sample_rate': getattr(args, 'sample_rate', 16000),
            'log_mel': getattr(args, 'log_mel', True),
            'mode': 'TB-KWS',
            'speaker_encoder_path': getattr(args, 'speaker_encoder_path', 
                                            'model/speaker/eff_vad_tdnn_s_mel40_25ms.jit'),
            'freeze_speaker_encoder': True,
        }
    
    # 建立模型
    model = p_ukws.P_UKWS(**config)
    
    # 載入權重
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # Handle key remapping between different FiLM naming conventions
    # Some checkpoints use 'enhanced_film.*' while model expects 'soft_film.*' or vice versa
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    
    # Check if remapping is needed
    needs_remap = False
    remap_from, remap_to = None, None
    
    if any('enhanced_film' in k for k in ckpt_keys) and any('soft_film' in k for k in model_keys):
        needs_remap = True
        remap_from, remap_to = 'enhanced_film', 'soft_film'
        print(f"⚠️ Key remapping: checkpoint uses 'enhanced_film' but model expects 'soft_film'")
    elif any('soft_film' in k for k in ckpt_keys) and any('enhanced_film' in k for k in model_keys):
        needs_remap = True
        remap_from, remap_to = 'soft_film', 'enhanced_film'
        print(f"⚠️ Key remapping: checkpoint uses 'soft_film' but model expects 'enhanced_film'")
    
    if needs_remap:
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace(remap_from, remap_to)
            new_state_dict[new_key] = v
        state_dict = new_state_dict
    
    # Load with strict=False to handle any remaining mismatches gracefully
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"⚠️ Missing keys (will use random init): {missing_keys[:5]}..." if len(missing_keys) > 5 else f"⚠️ Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"⚠️ Unexpected keys (ignored): {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"⚠️ Unexpected keys: {unexpected_keys}")
    
    print(f"✅ Model loaded from {checkpoint_path}")
    print(f"   Config: audio_input={config['audio_input']}, text_input={config['text_input']}, "
          f"stack_extractor={config['stack_extractor']}")
    return model, config


def create_dataloader(pkl_path, config, args):
    """
    建立資料載入器
    
    Args:
        pkl_path: Path to pkl file
        config: Config dict from load_model (contains audio_input, text_input, etc.)
        args: Command line args (for batch_size, max_samples)
    """
    # Google embedding directory
    if config['audio_input'] == "raw":
        gemb_dir = None
    else:
        gemb_dir = '/padawan/google_speech_embedding/DB/'
    
    dataset = personalized_libriphrase.PersonalizedLibriPhraseDataset(
        batch_size=args.batch_size,
        gemb_dir=gemb_dir,
        features=config['text_input'],
        train=False,  # Validation mode
        types='both',  # Load both easy and hard
        shuffle=False,
        pkl=pkl_path,
        frame_length=config['frame_length'],
        hop_length=config['hop_length'],
        personalized=True,  # Use personalized mode
        speaker_ratio=0.5,
        max_samples=getattr(args, 'max_samples', None),
    )
    
    loader = KWSDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate,
        pin_memory=True,
    )
    
    return loader


def main():
    parser = argparse.ArgumentParser(description='P_phon Detailed Analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--val_pkl', type=str, nargs='+', required=True,
                        help='Path(s) to validation PKL file(s). Multiple paths allowed for Easy/Hard.')
    parser.add_argument('--output_dir', type=str, default='analysis_results/',
                        help='Output directory')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples per dataset (for quick testing)')
    
    # Model config
    parser.add_argument('--audio_input', type=str, default='both',
                        choices=['raw', 'google_embed', 'both'])
    parser.add_argument('--text_input', type=str, default='g2p_embed')
    parser.add_argument('--stack_extractor', type=int, default=1)
    parser.add_argument('--speaker_encoder_path', type=str, 
                        default=None,
                        help='Override speaker encoder path (default: use path from checkpoint)')
    
    # NEW: t-SNE visualization option
    parser.add_argument('--tsne', action='store_true',
                        help='Enable t-SNE embedding visualization (slower but insightful)')
    parser.add_argument('--tsne_max_samples', type=int, default=5000,
                        help='Maximum samples for t-SNE (more samples = slower)')
    
    args = parser.parse_args()
    
    # 設定 device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 載入模型 (also returns config from checkpoint)
    model, config = load_model(args.checkpoint, args)
    model = model.to(device)
    model.eval()
    
    # 建立分析器
    analyzer = PPhonAnalyzer(args.output_dir)
    
    # 處理每個 PKL 檔案
    # NOTE: 如果只有一個 PKL 且 types='both'，則包含 easy 和 hard
    # difficulty 會在 batch 層級透過 batch['type'] 正確判斷
    
    for i, pkl_path in enumerate(args.val_pkl):
        # 判斷是單一 PKL (mixed) 還是分開的 easy/hard PKL
        if len(args.val_pkl) == 1:
            # 單一 PKL，包含 easy 和 hard，difficulty 會從 batch['type'] 提取
            difficulty_label = 'mixed (easy+hard)'
            tqdm_desc = 'Analyzing (auto-detect difficulty)'
        else:
            # 多個 PKL，按順序假設為 easy, hard
            difficulty_names = ['easy', 'hard']
            difficulty_label = difficulty_names[i] if i < len(difficulty_names) else f'loader_{i}'
            tqdm_desc = f'Analyzing {difficulty_label}'
        
        print(f"\n{'='*60}")
        print(f"Processing: {pkl_path}")
        if len(args.val_pkl) == 1:
            print(f"  📁 Single PKL mode: difficulty will be auto-detected from batch['type']")
        else:
            print(f"  (difficulty={difficulty_label})")
        if args.tsne:
            print(f"  📊 t-SNE embedding collection ENABLED")
        print(f"{'='*60}")
        
        loader = create_dataloader(pkl_path, config, args)
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=tqdm_desc):
                # Move to device
                batch_device = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch_device[k] = v.to(device)
                    else:
                        batch_device[k] = v
                
                # Prepare inputs (using config from checkpoint)
                if config['audio_input'] == "raw":
                    speech_input = batch_device["x"]
                    speech_len = batch_device["x_len"]
                elif config['audio_input'] == "google_embed":
                    speech_input = batch_device["gemb"]
                    speech_len = batch_device["gemb_len"]
                elif config['audio_input'] == "both":
                    speech_input = (batch_device["x"], batch_device["gemb"])
                    speech_len = (batch_device["x_len"], batch_device["gemb_len"])
                
                # Forward pass (with optional embedding extraction for t-SNE)
                output = model(
                    speech=speech_input,
                    text=batch_device["y"],
                    speech_len=speech_len,
                    text_len=batch_device["y_len"],
                    enrollment_audio=batch_device.get("enrollment_audio"),
                    return_embeddings=args.tsne,  # NEW: Enable embedding return for t-SNE
                )
                
                # Collect
                # Note: difficulty is auto-detected from batch['type'] in collect()
                # The fallback value only matters if batch['type'] is not available
                fallback_difficulty = 'unknown' if len(args.val_pkl) == 1 else difficulty_label
                analyzer.collect(output, batch_device, difficulty=fallback_difficulty)
    
    # 合併數據
    analyzer.finalize()
    
    # 儲存結果
    analyzer.save_results()
    
    print(f"\n✅ Analysis complete. Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
