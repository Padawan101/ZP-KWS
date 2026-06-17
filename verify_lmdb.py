import lmdb
import pickle
import numpy as np
import os

# 設定您想檢查的 LMDB 路徑列表
LMDB_PATHS = [
    '/home/DB/lmdb_train_500h',
    '/home/DB/lmdb_gemb_500h'
]

def verify(path):
    print(f"Checking: {path} ...")
    if not os.path.exists(path):
        print("❌ Path not found!")
        return

    try:
        env = lmdb.open(path, readonly=True, lock=False)
        with env.begin() as txn:
            # 1. 檢查總數量
            stats = txn.stat()
            print(f"   - Total entries: {stats['entries']}")
            
            # 2. 讀取第一筆資料看看 Key 和 Value
            cursor = txn.cursor()
            if cursor.first():
                key, value = cursor.item()
                key_str = key.decode('ascii')
                data = pickle.loads(value)
                
                print(f"   - Sample Key: {key_str}")
                print(f"   - Data Type: {type(data)}")
                
                if isinstance(data, np.ndarray):
                    print(f"   - Data Shape: {data.shape}")
                    print(f"   - Data Dtype: {data.dtype}")
                
                print("✅ LMDB seems OK!")
            else:
                print("❌ LMDB is empty!")
        env.close()
    except Exception as e:
        print(f"❌ Error opening LMDB: {e}")

if __name__ == "__main__":
    for p in LMDB_PATHS:
        verify(p)
        print("-" * 20)