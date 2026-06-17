"""
Alignment-Guided Contrastive (AGC) - Frame-level Mismatch Detection

計算 negative pairs 中，audio 的真實音素與 text 期望音素不匹配的 frame 位置。
用於在 Aux CE Loss 中對這些位置加權，迫使 Audio Encoder 更精準區分易混淆音素。
"""

import torch
import torch.nn.functional as F


def compute_frame_mismatch_mask(
    frame_labels: torch.Tensor,    # [B, T_a] MFA frame labels (42-class)
    text_indices: torch.Tensor,    # [B, T_t] G2P text phoneme indices (72-class)
    text_lens: torch.Tensor,       # [B] actual text lengths
    g2p_to_mfa: torch.LongTensor,  # [72] mapping G2P → MFA index
    z_keyword: torch.Tensor,       # [B] or [B,1] keyword labels
    pad_idx: int = 0,
    sil_idx: int = 1,
    spn_idx: int = 2,
) -> torch.Tensor:
    """
    計算 negative pairs 中每個 audio frame 的 mismatch mask。
    
    邏輯：
    1. 將 text phoneme 序列轉換到 MFA ID space（strip stress）
    2. 等比例展開 text 到 audio frame 長度
    3. 逐 frame 比對 audio 真實音素 vs text 期望音素
    4. 只標記「有意義的」不匹配（排除 PAD/SIL/SPN）
    
    只處理 negative pairs（positive pairs 的 mask 全為 0）。
    
    Args:
        frame_labels: [B, T_a] - 每個 audio frame 的真實音素 ID (MFA space)
        text_indices: [B, T_t] - text 的音素序列 ID (G2P space，含 stress)
        text_lens: [B] - text 實際長度（不含 padding），dtype 可能是 int32
        g2p_to_mfa: [V_g2p] - G2P index → MFA index lookup table
        z_keyword: [B] or [B,1] - 1=positive pair, 0=negative pair
        
    Returns:
        mismatch_mask: [B, T_a] BoolTensor
            True = 此 frame 的 audio 音素與 text 期望音素不同（且為 negative pair）
    """
    B, T_a = frame_labels.shape
    device = frame_labels.device
    mismatch_mask = torch.zeros(B, T_a, dtype=torch.bool, device=device)
    
    # 確保 z_keyword 是 [B]（防止 [B, 1] 造成 indexing 問題）
    z_kw = z_keyword.squeeze()
    if z_kw.dim() == 0:
        z_kw = z_kw.unsqueeze(0)
    
    # 防禦性檢查：確保 g2p_to_mfa 在正確 device 上
    # 通常已在初始化時移到 GPU，這裡是為了容錯
    if g2p_to_mfa.device != device:
        g2p_to_mfa = g2p_to_mfa.to(device)
    
    # 只處理 negative pairs
    is_neg = (z_kw == 0)
    if not is_neg.any():
        return mismatch_mask
    
    for i in range(B):
        if not is_neg[i]:
            continue
        
        # 1. 取得 text 音素序列（轉換到 MFA space）
        t_len = int(text_lens[i].item())
        if t_len <= 0:
            continue
        
        text_seq_g2p = text_indices[i, :t_len].long()  # [T_t_actual] G2P indices
        
        # Clamp to valid range for g2p_to_mfa lookup
        text_seq_g2p = text_seq_g2p.clamp(0, g2p_to_mfa.shape[0] - 1)
        text_seq_mfa = g2p_to_mfa[text_seq_g2p]  # [T_t_actual] MFA indices
        
        # 2. 等比例展開 text 到 audio frame 長度
        # 用 nearest interpolation: [1, 1, T_t] → [1, 1, T_a]
        expanded = F.interpolate(
            text_seq_mfa.float().unsqueeze(0).unsqueeze(0),
            size=T_a,
            mode='nearest'
        ).squeeze().long()  # [T_a]
        
        # 3. 比對 mismatch
        audio_phones = frame_labels[i]  # [T_a] MFA indices
        is_different = (audio_phones != expanded)
        
        # 4. 排除不可靠的 frames
        # audio 側：PAD、SIL、SPN 不計較
        audio_valid = (audio_phones != pad_idx) & (audio_phones != sil_idx) & (audio_phones != spn_idx)
        # text 側：PAD、SIL、SPN 也不計較（空格映射到 SIL）
        text_valid = (expanded != pad_idx) & (expanded != sil_idx) & (expanded != spn_idx)
        
        mismatch_mask[i] = is_different & audio_valid & text_valid
    
    return mismatch_mask
