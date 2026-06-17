#!/usr/bin/env python3
"""
textgrid_to_framelabels.py (Fixed Version using tgt)

將 MFA TextGrid 輸出轉換為 frame-level phoneme labels，儲存為 LMDB。
使用 tgt 套件進行穩健解析，解決 duration 讀取錯誤問題。

"""

import argparse
import json
import lmdb
import math
import os
import pickle
from pathlib import Path
from collections import Counter
import numpy as np
import tgt  # 核心修正：使用 tgt 套件

# Frame shift in seconds (20ms)
FRAME_SHIFT_SEC = 0.020

# Default vocabulary
DEFAULT_PHONEME_VOCAB = {
    "<PAD>": 0, "SIL": 1, "SPN": 2,
    "AA": 3, "AE": 4, "AH": 5, "AO": 6, "AW": 7, "AY": 8,
    "B": 9, "CH": 10, "D": 11, "DH": 12,
    "EH": 13, "ER": 14, "EY": 15,
    "F": 16, "G": 17, "HH": 18,
    "IH": 19, "IY": 20, "JH": 21, "K": 22, "L": 23,
    "M": 24, "N": 25, "NG": 26,
    "OW": 27, "OY": 28, "P": 29, "R": 30, "S": 31, "SH": 32,
    "T": 33, "TH": 34, "UH": 35, "UW": 36,
    "V": 37, "W": 38, "Y": 39, "Z": 40, "ZH": 41,
}

# Fallback mapping
DEFAULT_IPA_TO_ARPABET = {
    "spn": "SPN", "sil": "SIL", "": "SIL"
}

def parse_textgrid_tgt(textgrid_path: str, ipa_to_arpabet: dict, arpabet_vocab: dict):
    """
    使用 tgt 解析 TextGrid，返回 intervals 和 total_duration
    """
    try:
        # include_empty_intervals=True 確保我們能抓到 silence
        tg = tgt.read_textgrid(textgrid_path, include_empty_intervals=True)
    except Exception as e:
        print(f"Error reading {textgrid_path}: {e}")
        return [], 0.0

    # 獲取 Phones tier
    phones_tier = tg.get_tier_by_name("phones")
    
    # 總時長直接從 TextGrid 物件獲取，這是最準確的
    total_duration = tg.end_time
    
    parsed_intervals = []
    for interval in phones_tier:
        ipa = interval.text.strip()
        
        # Mapping
        arpabet = ipa_to_arpabet.get(ipa, "SIL")
        phone_id = arpabet_vocab.get(arpabet, arpabet_vocab["SIL"])
        
        parsed_intervals.append({
            "start_sec": interval.start_time,
            "end_sec": interval.end_time,
            "phone_id": phone_id,
            "arpabet": arpabet
        })
        
    return parsed_intervals, total_duration

def intervals_to_frame_labels(intervals: list, total_duration_sec: float) -> np.ndarray:
    """轉換 time intervals 為 frame indices"""
    n_frames = math.ceil(total_duration_sec / FRAME_SHIFT_SEC)
    
    # 防止空檔案導致崩潰，至少給 1 frame (SIL)
    if n_frames == 0:
        return np.array([1], dtype=np.int64)

    labels = np.ones(n_frames, dtype=np.int64) * 1  # 預設填 SIL (1)
    
    for interval in intervals:
        start_frame = math.floor(interval["start_sec"] / FRAME_SHIFT_SEC)
        end_frame = math.ceil(interval["end_sec"] / FRAME_SHIFT_SEC)
        
        # 邊界保護
        start_frame = max(0, min(start_frame, n_frames))
        end_frame = max(0, min(end_frame, n_frames))
        
        if end_frame > start_frame:
            labels[start_frame:end_frame] = interval["phone_id"]
            
    return labels

def extract_file_id(textgrid_path: str) -> str:
    return os.path.basename(textgrid_path).replace('.TextGrid', '')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--textgrid_dir", type=str, required=True)
    parser.add_argument("--output_lmdb", type=str, required=True)
    parser.add_argument("--mapping_json", type=str, default=None)
    parser.add_argument("--map_size_gb", type=int, default=10)
    args = parser.parse_args()

    # Load mapping
    if args.mapping_json and os.path.exists(args.mapping_json):
        with open(args.mapping_json, 'r') as f:
            mapping = json.load(f)
        ipa_to_arpabet = mapping.get("ipa_to_arpabet", DEFAULT_IPA_TO_ARPABET)
        arpabet_vocab = mapping.get("arpabet_vocab", DEFAULT_PHONEME_VOCAB)
        print(f"Loaded mapping from JSON. Vocab size: {len(arpabet_vocab)}")
    else:
        print("Using default mapping.")
        ipa_to_arpabet = DEFAULT_IPA_TO_ARPABET
        arpabet_vocab = DEFAULT_PHONEME_VOCAB

    # Scan files
    textgrid_files = list(Path(args.textgrid_dir).glob("**/*.TextGrid"))
    print(f"Found {len(textgrid_files)} TextGrid files.")

    # Open LMDB
    map_size = args.map_size_gb * 1024**3
    env = lmdb.open(args.output_lmdb, map_size=map_size)
    
    success = 0
    frame_lens = []
    
    with env.begin(write=True) as txn:
        # Store Metadata
        txn.put(b"__phoneme_vocab__", pickle.dumps(arpabet_vocab))
        txn.put(b"__n_phonemes__", pickle.dumps(len(arpabet_vocab)))
        txn.put(b"__frame_shift_sec__", pickle.dumps(FRAME_SHIFT_SEC))

        for i, tg_path in enumerate(textgrid_files):
            try:
                # 使用新的解析函數
                intervals, duration = parse_textgrid_tgt(str(tg_path), ipa_to_arpabet, arpabet_vocab)
                
                # 轉 Label
                labels = intervals_to_frame_labels(intervals, duration)
                
                # 寫入
                file_id = extract_file_id(str(tg_path))
                txn.put(file_id.encode(), pickle.dumps(labels))
                
                success += 1
                frame_lens.append(len(labels))
                
                if (i+1) % 10000 == 0:
                    print(f"Processed {i+1} files...")
            except Exception as e:
                print(f"Failed {tg_path}: {e}")

    print(f"\nDone! Success: {success}")
    if frame_lens:
        print(f"Avg frames: {np.mean(frame_lens):.1f}")
        print(f"Max frames: {max(frame_lens)}")

if __name__ == "__main__":
    main()