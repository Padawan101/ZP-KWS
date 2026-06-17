import lmdb
import pandas as pd
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import pickle
import os
import argparse

def save_to_lmdb(csv_path, lmdb_path, wav_root):
    print(f"Reading CSV: {csv_path}")
    print(f"WAV Root Dir: {wav_root}")
    
    df = pd.read_csv(csv_path)
    
    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path)

    # 預估 map_size (1TB 虛擬上限)
    env = lmdb.open(lmdb_path, map_size=1099511627776)
    
    success_count = 0
    error_count = 0

    # --- [修正] 手動開啟第一個交易 ---
    txn = env.begin(write=True)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # --- 內部函式：處理路徑與讀取 ---
            def process_path(rel_path):
                # 嘗試 1: 直接串接
                p1 = os.path.join(wav_root, rel_path)
                if os.path.exists(p1): return p1
                
                # 嘗試 2: 移除第一層目錄 (修復重複路徑問題)
                parts = rel_path.split(os.sep)
                if len(parts) > 1:
                    p2 = os.path.join(wav_root, *parts[1:])
                    if os.path.exists(p2): return p2
                
                # 嘗試 3: 如果是 .wav 找不到，試試 .flac
                if rel_path.endswith('.wav'):
                    rel_path_flac = rel_path.replace('.wav', '.flac')
                    p3 = os.path.join(wav_root, rel_path_flac)
                    if os.path.exists(p3): return p3
                    
                    # 針對嘗試 2 的邏輯也做一次 .flac 檢查
                    if len(parts) > 1:
                        p4 = os.path.join(wav_root, *parts[1:]).replace('.wav', '.flac')
                        if os.path.exists(p4): return p4

                raise FileNotFoundError(f"Cannot find file: {rel_path} in {wav_root}")

            # --- 處理 Anchor ---
            real_anchor_path = process_path(row['anchor'])
            fs, audio_data = wavfile.read(real_anchor_path)
            
            # Key: 使用 CSV 裡的原始路徑
            key_anchor = row['anchor'].encode('ascii')
            txn.put(key_anchor, pickle.dumps(audio_data))
            
            # --- 處理 Comparison ---
            real_comp_path = process_path(row['comparison'])
            fs2, audio_data2 = wavfile.read(real_comp_path)
            
            key_comp = row['comparison'].encode('ascii')
            txn.put(key_comp, pickle.dumps(audio_data2))
            
            success_count += 1

        except Exception as e:
            if error_count < 10:
                print(f"[Error] Row {idx}: {e}")
            error_count += 1
            continue

        # --- [修正] 每 5000 筆 Commit 一次 ---
        # 注意：要避開 idx=0，否則一開始就會 commit 空的交易
        if idx > 0 and idx % 5000 == 0:
            txn.commit()
            txn = env.begin(write=True)
    
    # --- [修正] 迴圈結束後，提交最後一批資料 ---
    txn.commit()
    env.close()
        
    print(f"Finished! Success: {success_count}, Errors: {error_count}")
    print(f"LMDB saved at: {lmdb_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='Path to input CSV')
    parser.add_argument('--out', type=str, required=True, help='Path to output LMDB directory')
    parser.add_argument('--wav_root', type=str, required=True, help='Root directory containing the wav files')
    args = parser.parse_args()
    
    save_to_lmdb(args.csv, args.out, args.wav_root)