"""
G2P → MFA Phoneme ID Mapping

G2P (libriphrase.py) 使用 72 類 stressed ARPAbet：
  <pad>=0, AA0=1, AA1=2, AA2=3, ..., ' '=71

MFA (frame_labels.lmdb) 使用 42 類 stress-free ARPAbet：
  PAD=0, SIL=1, SPN=2, AA=3, AE=4, ..., ZH=41

此模組建立 lookup table 將 G2P index 映射到 MFA index，
用於 frame-level mismatch detection。
"""

import re
import torch


# 預設 MFA 42-class vocab（如果 LMDB 沒有存）
MFA_VOCAB = {p: i for i, p in enumerate(['PAD', 'SIL', 'SPN'] + [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH'
])}


def build_g2p_to_mfa_mapping(g2p_phonemes: list, mfa_vocab: dict) -> torch.LongTensor:
    """
    建立 G2P index → MFA index 的映射表。
    
    Args:
        g2p_phonemes: list of str, G2P 音素列表（按 index 排序）
            例如 ['<pad>', 'AA0', 'AA1', 'AA2', ..., ' ']
            取自 libriphrase.py 的 phonemes 變數
            
        mfa_vocab: dict, {phoneme_string: mfa_index}
            例如 {'PAD': 0, 'SIL': 1, 'SPN': 2, 'AA': 3, ...}
            取自 frame_labels.lmdb 的 __phoneme_vocab__ 或 config
    
    Returns:
        mapping: torch.LongTensor of shape [len(g2p_phonemes)]
            mapping[g2p_idx] = mfa_idx
    """
    mapping = torch.zeros(len(g2p_phonemes), dtype=torch.long)
    unmapped = []
    mapped_samples = []  # 記錄映射樣本供 debug
    
    for g2p_idx, phoneme_str in enumerate(g2p_phonemes):
        # Special tokens
        if phoneme_str == '<pad>':
            mapping[g2p_idx] = mfa_vocab.get('PAD', 0)
            continue
        if phoneme_str == ' ':
            # G2P 的空格 = 詞邊界 → 對應 MFA 的 SIL
            mapping[g2p_idx] = mfa_vocab.get('SIL', 1)
            continue
        
        # Strip stress digit: AA0 → AA, AA1 → AA, B → B
        base_phoneme = re.sub(r'[012]$', '', phoneme_str)
        
        # Lookup in MFA vocab
        if base_phoneme in mfa_vocab:
            mapping[g2p_idx] = mfa_vocab[base_phoneme]
            # 記錄前幾個映射樣本
            if len(mapped_samples) < 10:
                mapped_samples.append(f"G2P[{g2p_idx}]='{phoneme_str}' → MFA[{mfa_vocab[base_phoneme]}]='{base_phoneme}'")
        else:
            # Fallback: map to PAD (will be ignored)
            mapping[g2p_idx] = 0
            unmapped.append(f"G2P[{g2p_idx}]='{phoneme_str}' → '{base_phoneme}' not in MFA vocab")
    
    # Debug output
    print(f"\n📍 [G2P→MFA Mapping Debug]")
    print(f"   Input G2P phonemes: {len(g2p_phonemes)}")
    print(f"   MFA vocab size: {len(mfa_vocab)}")
    print(f"   Mapping samples:")
    for sample in mapped_samples[:5]:
        print(f"     {sample}")
    
    if unmapped:
        print(f"⚠️  [Phoneme Mapping] {len(unmapped)} unmapped phonemes:")
        for msg in unmapped[:5]:
            print(f"    {msg}")
    else:
        print(f"✓ [Phoneme Mapping] All {len(g2p_phonemes)} G2P phonemes mapped to MFA space")
    
    return mapping


def get_g2p_phonemes() -> list:
    """
    回傳 G2P 音素列表（與 libriphrase.py 中的定義完全一致）。
    """
    return ["<pad>"] + [
        'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
        'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH',
        'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
        'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2',
        'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0',
        'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1',
        'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH',
        ' '
    ]


def load_mfa_vocab(frame_labels_path: str) -> dict:
    """
    從 frame_labels.lmdb 載入 MFA 音素 vocab。
    
    Returns:
        dict: {phoneme_string: mfa_index}
    """
    import lmdb
    import pickle
    
    env = lmdb.open(frame_labels_path, readonly=True, lock=False)
    with env.begin(write=False) as txn:
        vocab_bytes = txn.get(b'__phoneme_vocab__')
        if vocab_bytes is not None:
            return pickle.loads(vocab_bytes)
    
    # Fallback: 如果 LMDB 沒有存 vocab，用預設的 42 類
    print("⚠️  [MFA Vocab] __phoneme_vocab__ not found in LMDB, using default 42-class vocab")
    default_phonemes = ['PAD', 'SIL', 'SPN'] + [
        'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
        'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
        'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
        'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH'
    ]
    return {p: i for i, p in enumerate(default_phonemes)}
