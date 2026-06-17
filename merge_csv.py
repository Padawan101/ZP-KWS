import pandas as pd
import glob
import os

# 定義路徑 (對應 Docker 內部的路徑)
BASE_DIR = '/home/DB/LibriPhrase/data'

def merge_files(subset_name, output_name):
    # 搜尋類似 train_100h_pairs_1word.csv, 2word.csv ... 的檔案
    pattern = os.path.join(BASE_DIR, f'{subset_name}_*word.csv')
    all_files = glob.glob(pattern)
    
    if not all_files:
        print(f"[Warning] No files found for pattern: {pattern}")
        return

    print(f"Found {len(all_files)} files for {subset_name}. Merging...")
    # 讀取並合併
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    
    # 儲存
    output_path = os.path.join(BASE_DIR, output_name)
    df.to_csv(output_path, index=False)
    print(f"Saved merged CSV to: {output_path}")

if __name__ == "__main__":
    # 合併 100h
    merge_files('train_100h_pairs', 'train_100h_FULL.csv')
    
    # 合併 360h
    merge_files('train_360h_pairs', 'train_360h_FULL.csv')
    
    # 合併 500h
    merge_files('train_500h_pairs', 'train_500h_FULL.csv')