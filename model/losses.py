"""
Focal Loss implementations for PhonMatchNet Ablation Study

This module provides Focal Loss variants for:
- B2: Focal Loss on L_phon (phoneme-level matching)
- B3: Focal Loss on L_utt (utterance-level detection)

Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling hard samples (B3: L_utt)
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        gamma: focusing parameter (default: 2.0)
               - γ = 0: 等同於標準 BCE
               - γ = 2: 常用值
               - γ = 5: 極端關注 hard samples
        alpha: class balance weight (default: None, no balancing)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, gamma: float = 2.0, alpha: float = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: predicted probabilities (after sigmoid), shape [B] or [B, 1]
            target: ground truth labels (0 or 1), shape [B] or [B, 1]
        
        Returns:
            Focal loss scalar or per-sample loss depending on reduction
        """
        # Flatten if needed
        pred = pred.view(-1)
        target = target.view(-1).float()
        
        # Clamp for numerical stability
        pred = pred.clamp(min=1e-7, max=1 - 1e-7)
        
        # BCE (no reduction)
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        
        # p_t: probability of correct class
        p_t = torch.where(target == 1, pred, 1 - pred)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weight (optional)
        if self.alpha is not None:
            alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
            focal_weight = focal_weight * alpha_t
        
        loss = focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class PhonemeAwareFocalLoss(nn.Module):
    """
    Focal Loss for phoneme-level predictions (B2: L_phon)
    
    關鍵修正：使用 per-phoneme paired labels（text == speech），
    而非 broadcast utterance label。行為與 sequence_cross_entropy 完全一致。
    
    Args:
        gamma: focusing parameter (default: 2.0)
        alpha: class balance weight (default: None)
    """
    
    def __init__(self, gamma: float = 2.0, alpha: float = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(
        self, 
        speech_label: torch.Tensor,      # [B, Ls] phoneme indices
        text_label: torch.Tensor,        # [B, Lt] phoneme indices
        logits: torch.Tensor,            # [B, Lt] raw logits (before sigmoid)
        logits_mask: torch.Tensor,       # [B, Lt] valid positions
    ) -> torch.Tensor:
        """
        行為與 sequence_cross_entropy 完全一致，只是把 BCE 換成 Focal。
        
        Args:
            speech_label: [B, Ls] - speech phoneme indices
            text_label: [B, Lt] - text phoneme indices
            logits: [B, Lt] - raw logits (before sigmoid)
            logits_mask: [B, Lt] - valid positions mask
        
        Returns:
            loss: mean_per_token * B（與 sequence_cross_entropy reduction='sum' 一致）
        """
        B = logits.size(0)
        
        # === 與 sequence_cross_entropy 完全相同的 label 邏輯 ===
        if text_label.shape[1] > speech_label.shape[1]:
            speech_label = F.pad(
                speech_label, 
                (0, text_label.shape[1] - speech_label.shape[1]), 
                'constant', value=0
            )
        elif text_label.shape[1] < speech_label.shape[1]:
            speech_label = speech_label[:, :text_label.shape[1]]
        
        paired_label = torch.logical_and(
            text_label == speech_label, logits_mask
        ).float()
        
        # Flatten valid tokens
        paired_label = torch.masked_select(
            paired_label, logits_mask.bool()
        ).view(-1)
        logits_flat = torch.masked_select(
            logits, logits_mask.bool()
        ).view(-1)
        
        total_tokens = logits_flat.shape[0]
        if total_tokens == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # === Focal Loss 計算（使用 logits，數值更穩定）===
        # 先算 BCE
        bce = F.binary_cross_entropy_with_logits(
            logits_flat, paired_label, reduction='none'
        )
        
        # 算 p_t
        pred = torch.sigmoid(logits_flat)
        p_t = torch.where(paired_label == 1, pred, 1 - pred)
        
        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = torch.where(
                paired_label == 1, self.alpha, 1 - self.alpha
            )
            focal_weight = focal_weight * alpha_t
        
        loss = (focal_weight * bce).sum()
        
        # === 與 sequence_cross_entropy 一致的 normalization ===
        loss = loss / total_tokens  # mean per token
        loss = loss * B             # * batch_size
        
        return torch.nan_to_num(loss)


class FocalLossWithLogits(nn.Module):
    """
    Focal Loss that accepts logits (before sigmoid).
    
    Useful for L_utt when working with raw model outputs.
    
    Args:
        gamma: focusing parameter (default: 2.0)
        alpha: class balance weight (default: None)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, gamma: float = 2.0, alpha: float = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: raw logits (before sigmoid), shape [B] or [B, 1]
            target: ground truth labels (0 or 1), shape [B] or [B, 1]
        
        Returns:
            Focal loss
        """
        # Flatten if needed
        logits = logits.view(-1)
        target = target.view(-1).float()
        
        # Compute probabilities
        pred = torch.sigmoid(logits)
        pred = pred.clamp(min=1e-7, max=1 - 1e-7)
        
        # BCE with logits for numerical stability
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        
        # p_t: probability of correct class
        p_t = torch.where(target == 1, pred, 1 - pred)
        
        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weight (optional)
        if self.alpha is not None:
            alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
            focal_weight = focal_weight * alpha_t
        
        loss = focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class HardLabelWeightedPCL(nn.Module):
    """
    修正版：使用 per-phoneme paired labels 來計算 contrastive loss (B4)
    
    核心改變：
    - 不再對所有音素用同一個方向 push
    - 只對 negative 樣本中「不 match 但預測高」的音素做懲罰
    - 使用 paired_label 區分真正 mismatch 的音素位置
    
    Args:
        hard_weight: Hard negative 的權重倍數
    """
    
    def __init__(self, hard_weight: float = 2.0):
        super().__init__()
        self.hard_weight = hard_weight
    
    def forward(
        self,
        P_phon: torch.Tensor,            # [B, T_t] after sigmoid
        speech_label: torch.Tensor,      # [B, Ls]
        text_label: torch.Tensor,        # [B, Lt]
        z_keyword: torch.Tensor,         # [B] utterance label
        is_hard: torch.Tensor,           # [B] hard/easy flag
        phoneme_mask: torch.Tensor,      # [B, T_t]
    ) -> torch.Tensor:
        """
        Args:
            P_phon: [B, T_t] - phoneme predictions (after sigmoid)
            speech_label: [B, Ls] - speech phoneme indices
            text_label: [B, Lt] - text phoneme indices
            z_keyword: [B] - utterance-level keyword labels
            is_hard: [B] - hard/easy labels from LibriPhrase dataset
            phoneme_mask: [B, T_t] - valid phoneme positions
        """
        if z_keyword.dim() > 1:
            z_keyword = z_keyword.squeeze(-1)
        if is_hard.dim() > 1:
            is_hard = is_hard.squeeze(-1)
        
        B, T_t = P_phon.shape
        
        # 建立 per-phoneme paired labels（與 sequence_cross_entropy 一致）
        if text_label.shape[1] > speech_label.shape[1]:
            speech_label = F.pad(
                speech_label, 
                (0, text_label.shape[1] - speech_label.shape[1]),
                'constant', value=0
            )
        elif text_label.shape[1] < speech_label.shape[1]:
            speech_label = speech_label[:, :text_label.shape[1]]
        
        paired_label = torch.logical_and(
            text_label == speech_label, phoneme_mask
        ).float()  # [B, T_t]
        
        # === 正樣本：所有 match 的音素 → push P_phon 往 1 ===
        # === 負樣本：只有 mismatch 的音素 → push P_phon 往 0 ===
        # 對於 negative 的 matched phonemes，不施加任何力（它們本來就該高）
        
        is_pos = (z_keyword == 1).float().unsqueeze(1)  # [B, 1]
        is_neg = (z_keyword == 0).float().unsqueeze(1)  # [B, 1]
        mismatch_mask = (1 - paired_label) * phoneme_mask.float()  # [B, T_t]
        
        # Positive loss: all valid phonemes → push to 1
        pos_loss = is_pos * phoneme_mask.float() * (1 - P_phon) ** 2
        
        # Negative loss: only mismatched phonemes → push to 0
        neg_loss = is_neg * mismatch_mask * P_phon ** 2
        
        # Per-sample mean
        pos_count = (is_pos * phoneme_mask.float()).sum(dim=1).clamp(min=1)
        neg_count = (is_neg * mismatch_mask).sum(dim=1).clamp(min=1)
        
        sample_loss = pos_loss.sum(dim=1) / pos_count + neg_loss.sum(dim=1) / neg_count
        
        # Hard negative weighting
        weight = torch.ones(B, device=P_phon.device)
        hard_neg_mask = (z_keyword == 0) & is_hard.bool()
        weight[hard_neg_mask] = self.hard_weight
        
        return (sample_loss * weight).sum()


class DynamicHardnessPCL(nn.Module):
    """
    P1: Dynamic Hardness PCL
    
    用模型自身的預測值（P_utt）動態判斷 hardness，
    取代 dataset 的靜態 hard/easy 標籤。
    
    原理：
    - Negative 樣本的 P_utt 越高 → 越容易被誤判 → 權重越大
    - Positive 樣本的 P_utt 越低 → 越容易漏掉 → 權重越大
    
    Score 計算（簡化版）：
    - 統一用 min(P_phon) 作為 score
    - KWS 是 "AND" 邏輯：一個音素錯就該被拒絕
    - 這樣不需要計算 paired_label / mismatch_mask
    """
    
    def __init__(self, hard_scale: float = 3.0):
        super().__init__()
        self.hard_scale = hard_scale
    
    def forward(
        self,
        P_phon: torch.Tensor,            # [B, T_t] after sigmoid
        P_utt: torch.Tensor,             # [B] or [B,1] after sigmoid
        speech_label: torch.Tensor,      # [B, Ls] (unused, kept for interface)
        text_label: torch.Tensor,        # [B, Lt] (unused, kept for interface)
        z_keyword: torch.Tensor,         # [B] utterance label
        phoneme_mask: torch.Tensor,      # [B, T_t]
    ) -> torch.Tensor:
        P_utt = P_utt.view(-1)           # [B]
        z_keyword = z_keyword.view(-1)    # [B]
        B, T_t = P_phon.shape
        
        is_pos = (z_keyword == 1)  # [B]
        is_neg = (z_keyword == 0)  # [B]
        
        # === 統一用 min(P_phon) 作為 score（KWS AND 邏輯）===
        # 對 invalid 位置設為 1.0 避免影響 min
        phon_for_min = P_phon.clone()
        phon_for_min[~phoneme_mask] = 1.0
        s = phon_for_min.min(dim=-1).values  # [B]
        
        # Positive: s → 1（min 越高越好）
        # Negative: s → 0（min 越低越好，代表模型能區分出差異）
        
        # === Base PCL loss（MSE style）===
        m = z_keyword.float()
        base_loss = m * (1 - s) ** 2 + (1 - m) * s ** 2  # [B]
        
        # === Dynamic Hardness Weight ===
        difficulty = torch.zeros(B, device=P_phon.device)
        difficulty[is_neg] = P_utt[is_neg].detach()       # high P_utt → hard neg
        difficulty[is_pos] = (1 - P_utt[is_pos]).detach() # low P_utt → hard pos
        
        weight = 1.0 + self.hard_scale * difficulty  # [B]
        
        return (base_loss * weight).sum()


class MarginPCL(nn.Module):
    """
    P2: Margin-based Asymmetric PCL
    
    取代 MSE 的 (1-s)² / s²，使用 margin hinge loss：
    - Positive: max(0, margin_pos - s)²    只在 s < margin_pos 時有梯度
    - Negative: max(0, s - margin_neg)²    只在 s > margin_neg 時有梯度
    
    好處：
    - 不強迫 s 逼到 0/1 極端值
    - margin 之內的樣本不產生梯度 → 減少無效更新
    - 模型可以把精力集中在 margin 邊界附近的難樣本
    
    內建 P1 的 Dynamic Hardness 功能（可選）
    """
    
    def __init__(
        self, 
        margin_pos: float = 0.8,
        margin_neg: float = 0.2,
        hard_scale: float = 3.0,
        use_dynamic_hardness: bool = True,
    ):
        """
        Args:
            margin_pos: positive 的目標下界（s 應 > margin_pos）
            margin_neg: negative 的目標上界（s 應 < margin_neg）
            hard_scale: dynamic hardness 的放大係數（P1 功能）
            use_dynamic_hardness: 是否啟用 P1 的 dynamic weighting
        """
        super().__init__()
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg
        self.hard_scale = hard_scale
        self.use_dynamic_hardness = use_dynamic_hardness
    
    def forward(
        self,
        P_phon: torch.Tensor,            # [B, T_t] after sigmoid
        P_utt: torch.Tensor,             # [B] or [B,1] after sigmoid
        speech_label: torch.Tensor,      # [B, Ls] (unused, kept for interface)
        text_label: torch.Tensor,        # [B, Lt] (unused, kept for interface)
        z_keyword: torch.Tensor,         # [B] utterance label
        phoneme_mask: torch.Tensor,      # [B, T_t]
    ) -> torch.Tensor:
        P_utt = P_utt.view(-1)
        z_keyword = z_keyword.view(-1)
        B, T_t = P_phon.shape
        
        is_pos = (z_keyword == 1)
        is_neg = (z_keyword == 0)
        
        # === 統一用 min(P_phon) 作為 score（KWS AND 邏輯）===
        phon_for_min = P_phon.clone()
        phon_for_min[~phoneme_mask] = 1.0
        s = phon_for_min.min(dim=-1).values  # [B]
        
        # === Margin Hinge Loss ===
        # Positive: s 應 > margin_pos
        # Negative: s 應 < margin_neg
        loss = torch.zeros(B, device=P_phon.device)
        
        if is_pos.any():
            loss[is_pos] = torch.clamp(self.margin_pos - s[is_pos], min=0) ** 2
        
        if is_neg.any():
            loss[is_neg] = torch.clamp(s[is_neg] - self.margin_neg, min=0) ** 2
        
        # === Dynamic Hardness Weight（可選，整合 P1）===
        if self.use_dynamic_hardness:
            difficulty = torch.zeros(B, device=P_phon.device)
            difficulty[is_neg] = P_utt[is_neg].detach()
            difficulty[is_pos] = (1 - P_utt[is_pos]).detach()
            weight = 1.0 + self.hard_scale * difficulty
        else:
            weight = torch.ones(B, device=P_phon.device)
        
        return (loss * weight).sum()


class PhonemePositionPCL(nn.Module):
    """
    P3: Confidence-Weighted Phoneme-Position Contrastive Loss (Optimized)
    
    不壓縮成 utterance-level scalar，直接在每個 phoneme 位置做 loss。
    
    修正點：
    1. 解決梯度消失：將 Base Loss 從 MSE (^2) 改為 L1 (^1)，
       避免 (1-p)^(gamma+2) = (1-p)^4 過度抑制中等難度樣本
    2. 統一 detach 行為：Weight 計算部分 detach，避免梯度混亂
    
    Loss 公式：
    - Positive: weight * error = (1-P)^gamma * (1-P) = (1-P)^(gamma+1)
    - Negative: weight * excess = P^gamma * max(0, P-margin)
    
    當 gamma=2 時，總體是 error^3 而非 error^4，梯度更健康。
    """
    
    def __init__(self, gamma: float = 2.0, margin_neg: float = 0.2):
        """
        Args:
            gamma: focusing parameter，越大越聚焦 hard phonemes
            margin_neg: negative mismatch 的 P_phon 目標上界
        """
        super().__init__()
        self.gamma = gamma
        self.margin_neg = margin_neg
    
    def forward(
        self,
        P_phon: torch.Tensor,            # [B, T_t] after sigmoid
        speech_label: torch.Tensor,      # [B, Ls]
        text_label: torch.Tensor,        # [B, Lt]
        z_keyword: torch.Tensor,         # [B] utterance label
        phoneme_mask: torch.Tensor,      # [B, T_t]
    ) -> torch.Tensor:
        z_keyword = z_keyword.view(-1)
        
        # === per-phoneme paired labels ===
        if text_label.shape[1] > speech_label.shape[1]:
            speech_label = F.pad(
                speech_label,
                (0, text_label.shape[1] - speech_label.shape[1]),
                'constant', value=0
            )
        elif text_label.shape[1] < speech_label.shape[1]:
            speech_label = speech_label[:, :text_label.shape[1]]
        
        paired_label = torch.logical_and(
            text_label == speech_label, phoneme_mask
        ).float()  # [B, T_t]
        
        valid_mask = phoneme_mask.float()
        mismatch_mask = (1 - paired_label) * valid_mask  # [B, T_t]
        
        is_pos = (z_keyword == 1).float().unsqueeze(1)  # [B, 1]
        is_neg = (z_keyword == 0).float().unsqueeze(1)  # [B, 1]
        
        # === Positive Loss: Push P_phon -> 1 ===
        # 原版: weight * (1-P)^2 -> (1-P)^(gamma+2) -> 梯度消失
        # 修正: weight * (1-P)   -> (1-P)^(gamma+1) -> 健康梯度
        pos_error = 1 - P_phon
        pos_weight = pos_error.detach() ** self.gamma  # Detach weight!
        pos_loss = is_pos * valid_mask * pos_weight * pos_error
        
        # === Negative Loss: Push P_phon < margin ===
        # 只對 mismatch 位置且 P_phon > margin 的施加 penalty
        excess = torch.clamp(P_phon - self.margin_neg, min=0)
        neg_weight = P_phon.detach() ** self.gamma  # Detach weight!
        neg_loss = is_neg * mismatch_mask * neg_weight * excess
        
        # === Per-sample normalization (Mean) ===
        pos_count = (is_pos * valid_mask).sum(dim=1).clamp(min=1)
        neg_count = (is_neg * mismatch_mask).sum(dim=1).clamp(min=1)
        
        sample_loss = pos_loss.sum(dim=1) / pos_count + neg_loss.sum(dim=1) / neg_count
        
        return sample_loss.sum()


# =============================================================================
# MFA Auxiliary CE Loss (Section 2-3 of MFA_AuxCE_Design.md)
# =============================================================================

def align_labels_to_encoder(frame_labels: torch.Tensor, encoder_output_len: int) -> torch.Tensor:
    """
    對齊 frame labels 長度與 Audio Encoder 輸出長度。
    
    Audio Encoder 輸出可能與 MFA 計算的 frame 數量有 ±1~2 差異（Conv padding 等因素）。
    
    Args:
        frame_labels: [T_mfa] 或 [B, T_mfa] - MFA frame-level phoneme labels
        encoder_output_len: Audio Encoder 實際輸出的 time dimension
        
    Returns:
        aligned_labels: 與 encoder_output_len 對齊後的 labels
    """
    if frame_labels.dim() == 1:
        label_len = frame_labels.shape[0]
        if label_len == encoder_output_len:
            return frame_labels
        elif label_len > encoder_output_len:
            # 截斷末尾
            return frame_labels[:encoder_output_len]
        else:
            # 末尾填充 PAD (0)
            pad_size = encoder_output_len - label_len
            return F.pad(frame_labels, (0, pad_size), value=0)
    else:
        # Batch mode: [B, T_mfa]
        B, label_len = frame_labels.shape
        if label_len == encoder_output_len:
            return frame_labels
        elif label_len > encoder_output_len:
            return frame_labels[:, :encoder_output_len]
        else:
            pad_size = encoder_output_len - label_len
            return F.pad(frame_labels, (0, pad_size), value=0)


class AuxPhonemeHead(nn.Module):
    """
    Auxiliary Phoneme Classification Head (Section 2.2)
    
    只用單層 Linear，不用 MLP：
    - 目的是約束 Audio Encoder 的 representation，不是建好分類器
    - 如果 head 太強，它自己學會分類，不逼 Audio Encoder 改善
    - 與 representation learning 中 "linear probe" 原理一致
    
    位置：接在 Audio Encoder 輸出上，FiLM 和 Self-Attention 之前
    推理時：完全移除，不增加任何計算
    """
    
    def __init__(self, feature_dim: int = 128, n_phonemes: int = 42):
        """
        Args:
            feature_dim: Audio Encoder 輸出維度 (default: 128)
            n_phonemes: 音素類別數 (default: 42 = PAD + SIL + SPN + 39 ARPAbet)
        """
        super().__init__()
        self.head = nn.Linear(feature_dim, n_phonemes)
        self.n_phonemes = n_phonemes
        
    def forward(self, E_a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            E_a: Audio Encoder 輸出 [B, T_a, feature_dim]
            
        Returns:
            logits: [B, T_a, n_phonemes] - 未經 softmax 的 logits
        """
        return self.head(E_a)


class AuxCELoss(nn.Module):
    """
    Auxiliary Cross-Entropy Loss (Section 2.3 + 3.6)
    
    包含：
    1. Base L_aux: 所有 frame 上的 phoneme classification CE loss
    2. Optional L_agc: Mismatch-weighted aux CE for negative pairs
    
    推薦權重:
    - λ_aux = 0.5 (search: 0.1, 0.3, 0.5, 1.0)
    - α_agc = 1.0 (mismatch frames 的額外權重)
    """
    
    def __init__(
        self, 
        n_phonemes: int = 42,
        ignore_index: int = 0,  # PAD
        alpha_agc: float = 1.0,
        label_smoothing: float = 0.1  # MFA 邊界噪音容錯
    ):
        """
        Args:
            n_phonemes: 音素類別數
            ignore_index: 忽略的 label index (PAD = 0)
            alpha_agc: L_agc mismatch 權重 (default: 1.0)
            label_smoothing: Label smoothing 係數 (default: 0.1)
        """
        super().__init__()
        self.n_phonemes = n_phonemes
        self.ignore_index = ignore_index
        self.alpha_agc = alpha_agc
        self.label_smoothing = label_smoothing
        self.ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index, 
            reduction='mean',
            label_smoothing=label_smoothing
        )
    
    def forward(
        self,
        aux_logits: torch.Tensor,           # [B, T_a, n_phonemes]
        frame_labels: torch.Tensor,          # [B, T_a]
        mismatch_mask: torch.Tensor = None,  # [B, T_a] BoolTensor, from compute_frame_mismatch_mask
    ) -> dict:
        """
        計算 Auxiliary CE Loss（含 frame-level AGC weighting）。
        
        當 mismatch_mask 提供時，mismatch frames 的 loss 會被放大 (1 + alpha_agc) 倍。
        這迫使 Audio Encoder 在「audio 和 text 音素不同」的位置更精準地識別音素。
        
        例如 "Bake" vs "Make"：
        - /ey/, /k/ 的 frames → 正常權重 (1.0)
        - /b/ vs /m/ 的 frames → 放大權重 (1 + alpha_agc)
        
        Args:
            aux_logits: [B, T_a, n_phonemes] - Aux head 輸出
            frame_labels: [B, T_a] - MFA frame-level phoneme labels
            mismatch_mask: [B, T_a] BoolTensor - True = mismatch frame（只在 negative pairs 中）
        """
        B, T_a, C = aux_logits.shape
        
        # 對齊長度
        frame_labels = align_labels_to_encoder(frame_labels, T_a)
        
        # Reshape for per-frame CE
        logits_flat = aux_logits.reshape(-1, C)    # [B*T_a, C]
        labels_flat = frame_labels.reshape(-1)      # [B*T_a]
        
        # 有效 frame mask（排除 PAD）
        valid = (labels_flat != self.ignore_index)
        
        if not valid.any():
            return {
                'loss': torch.tensor(0.0, device=aux_logits.device, requires_grad=True),
                'accuracy': torch.tensor(0.0, device=aux_logits.device),
                'n_mismatch': 0,
            }
        
        # 計算 per-frame CE loss（不 reduce）
        loss_per_frame = F.cross_entropy(
            logits_flat, labels_flat,
            ignore_index=self.ignore_index,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )  # [B*T_a]
        
        # 建立 frame 權重
        n_mismatch = 0
        if mismatch_mask is not None and mismatch_mask.any():
            # 對齊 mismatch_mask shape
            if mismatch_mask.shape[1] != T_a:
                if mismatch_mask.shape[1] > T_a:
                    mismatch_mask = mismatch_mask[:, :T_a]
                else:
                    mismatch_mask = F.pad(mismatch_mask, (0, T_a - mismatch_mask.shape[1]), value=False)
            
            mismatch_flat = mismatch_mask.reshape(-1).float()  # [B*T_a]
            weight = 1.0 + self.alpha_agc * mismatch_flat
            # matched frames: weight=1.0, mismatch frames: weight=1.0+alpha_agc
            
            n_mismatch = mismatch_mask.sum().item()
            
            # Weighted mean（只算 valid frames）
            weighted_loss = loss_per_frame * weight
            total_loss = weighted_loss[valid].sum() / weight[valid].sum()
        else:
            # 無 mismatch mask → 普通 mean CE
            total_loss = loss_per_frame[valid].mean()
        
        # Accuracy（監控用）
        with torch.no_grad():
            preds = logits_flat.argmax(dim=-1)
            
            # 1. 整體準確率（排除 PAD）
            if valid.any():
                accuracy = (preds[valid] == labels_flat[valid]).float().mean()
            else:
                accuracy = torch.tensor(0.0, device=aux_logits.device)
            
            # 2. 非靜音準確率 (Active Accuracy)
            # SIL=1, SPN=2 根據 MFA vocab 設定
            sil_mask = (labels_flat == 1) | (labels_flat == 2)
            active_mask = valid & (~sil_mask)  # 既不是 PAD 也不是 SIL/SPN
            
            if active_mask.any():
                acc_active = (preds[active_mask] == labels_flat[active_mask]).float().mean()
            else:
                acc_active = torch.tensor(0.0, device=aux_logits.device)
        
        return {
            'loss': total_loss,
            'accuracy': accuracy,
            'acc_active': acc_active,  # 非靜音準確率，用於檢測模型是否只猜 SIL
            'n_mismatch': n_mismatch,
        }


class AsymmetricMinPCL(nn.Module):
    """
    NegMinPCL v2: Asymmetric Min-Phoneme Contrastive Loss

    Negative 側：壓低 min(P_phon) 至 margin_neg 以下
    Positive 側（可選）：safety floor，只在 min(P_phon) 低於 margin_pos 時啟動

    與 v1 的差異：
    1. 新增 positive safety floor（default 啟用）
    2. Positive margin 設在遠低於 baseline 的位置（0.55），平時不干擾
    3. Positive 不使用 dynamic hardness（固定 weight=1.0），避免擾動

    Args:
        margin_neg: negative 的目標上界 (default: 0.2)
        margin_pos: positive 的安全底線 (default: 0.55)
        hard_scale: negative 側的 dynamic hardness scale (default: 3.0)
        enable_pos_floor: 是否啟用 positive safety floor (default: True)
    """

    def __init__(
        self,
        margin_neg: float = 0.2,
        margin_pos: float = 0.55,
        hard_scale: float = 3.0,
        enable_pos_floor: bool = True,
    ):
        super().__init__()
        self.margin_neg = margin_neg
        self.margin_pos = margin_pos
        self.hard_scale = hard_scale
        self.enable_pos_floor = enable_pos_floor

    def forward(
        self,
        P_phon: torch.Tensor,         # [B, T_t] after sigmoid
        P_utt: torch.Tensor,          # [B] or [B,1] after sigmoid
        z_keyword: torch.Tensor,      # [B] utterance label (0=neg, 1=pos)
        phoneme_mask: torch.Tensor,   # [B, T_t] valid phoneme positions
    ) -> torch.Tensor:
        P_utt = P_utt.detach().view(-1)
        z_keyword = z_keyword.view(-1)
        B, T_t = P_phon.shape

        is_neg = (z_keyword == 0)
        is_pos = (z_keyword == 1)

        # === 計算 min(P_phon) ===
        P_phon_masked = P_phon.clone()
        P_phon_masked[~phoneme_mask] = 1.0  # PAD → 不影響 min
        s = P_phon_masked.min(dim=-1).values  # [B]

        loss = torch.tensor(0.0, device=P_phon.device, requires_grad=True)

        # === Negative: 壓低 min(P_phon) 至 margin_neg 以下 ===
        if is_neg.any():
            n_neg = is_neg.sum().clamp(min=1)
            excess_neg = torch.clamp(s[is_neg] - self.margin_neg, min=0)
            base_neg = excess_neg ** 2

            # Dynamic hardness: P_utt 高的 negative 更危險
            difficulty = P_utt[is_neg]
            weight_neg = 1.0 + self.hard_scale * difficulty

            loss_neg = (base_neg * weight_neg).sum() / n_neg
            loss = loss + loss_neg

        # === Positive: safety floor（只在 min 低於底線時啟動）===
        if self.enable_pos_floor and is_pos.any():
            n_pos = is_pos.sum().clamp(min=1)
            deficit_pos = torch.clamp(self.margin_pos - s[is_pos], min=0)
            base_pos = deficit_pos ** 2

            # Positive 不用 dynamic hardness，固定權重
            loss_pos = base_pos.sum() / n_pos
            loss = loss + loss_pos

        return loss


# Backward-compatible alias
NegMinPCL = AsymmetricMinPCL

