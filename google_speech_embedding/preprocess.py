import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import wavfile
# 延遲 import 以避免 TF 初始化問題
# from speech_embedding import GoogleSpeechEmbedder 
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import glob
import soundfile as sf  # 確保有安裝 soundfile 以讀取 flac

fs = 16000
debug = False

def create_npy(data, wav_dir, save_dir, desc="libriphrase"):
    # 延遲 import
    from speech_embedding import GoogleSpeechEmbedder
    
    # data = data.sort_values(by='duration').reset_index(drop=True)
    wav_list = data['wav'].values
    
    # 計算最大長度
    if len(data) > 0:
        maxlen_a = int((int(data['duration'].values.max() / 0.5) + 1 ) * fs / 2)
    else:
        return

    EMB = GoogleSpeechEmbedder()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in tqdm(range(len(wav_list)), desc=desc):
        try:
            rel_path = wav_list[i]
            
            # --- 路徑搜尋邏輯 ---
            real_path = os.path.join(wav_dir, rel_path)
            
            # 處理重複目錄 (train-clean-100/train-clean-100)
            if not os.path.exists(real_path):
                parts = rel_path.split(os.sep)
                if len(parts) > 1:
                    real_path = os.path.join(wav_dir, *parts[1:])
            
            # 處理副檔名 (.wav -> .flac)
            if not os.path.exists(real_path) and real_path.endswith('.wav'):
                real_path = real_path.replace('.wav', '.flac')
            
            if not os.path.exists(real_path):
                # print(f"Skipping missing file: {rel_path}")
                continue
            # -------------------

            # 決定輸出路徑
            file_dirs = os.path.splitext(rel_path)[0]
            file_dirs = file_dirs.split("/")
            # 避免目錄過深
            if len(file_dirs) > 1 and file_dirs[0] == file_dirs[1]:
                file_dirs = file_dirs[1:]
            
            filename = file_dirs[-1] + '.npy'
            dirpath = os.path.join(*file_dirs[:-1])
            
            full_save_dir = os.path.join(save_dir, dirpath)
            full_save_path = os.path.join(full_save_dir, filename)
            
            if os.path.exists(full_save_path):
                continue
                
            if not os.path.exists(full_save_dir):
                os.makedirs(full_save_dir, exist_ok=True)
            
            # 讀取音檔
            try:
                audio_data, fs_read = sf.read(real_path)
            except Exception:
                # Fallback to scipy
                fs_read, audio_data = wavfile.read(real_path)

            x = [audio_data.astype(np.float32)] # soundfile 讀出來通常已經是 float 或需要正規化，這裡簡化處理
            # 如果是用 wavfile 讀取且不是 float，可能需要 / 32768.0
            # 但 GoogleSpeechEmbedder 通常預期 normalized input
            if audio_data.dtype == np.int16:
                 x = [audio_data.astype(np.float32) / 32768.0]
                 
            x = pad_sequences(np.array(x), maxlen=maxlen_a, value=0.0, padding='post', dtype=x[0].dtype)
            
            # 生成 Embedding
            emb = EMB(x).numpy()
            np.save(full_save_path, emb)

        except Exception as e:
            print(f"Error processing {wav_list[i]}: {e}")
            continue


def preprocess_libriphrase(wav_dir = '/home/DB/LibriPhrase_diffspk_all',
                           csv_dir = '/home/DB/LibriPhrase/data',
                           save_dir = '/home/google_speech_embedding/DB/LibriPhrase_diffspk_all',
                           train_csv = ['train_100h', 'train_360h'],
                           test_csv = ['train_500h',],
                           train = True,
                           ):

        print(f">> Processing LibriPhrase (Train={train})...")
        data_list = []

        for db in train_csv if train else test_csv:
                # 支援 FULL csv 或分散 csv
                full_csv = os.path.join(csv_dir, f"{db}_FULL.csv")
                if os.path.exists(full_csv):
                     csv_list = [full_csv]
                else:
                     csv_list = [str(x) for x in Path(csv_dir).rglob('*' + db + '*word*')]
                     if not csv_list:
                         csv_list = [str(x) for x in Path(csv_dir).rglob('*' + db + '*.csv')]

                for n_word in csv_list:
                        print(">> Reading CSV : {} ".format(n_word))
                        df = pd.read_csv(n_word)
                        
                        # --- [修正] 使用 concat 取代 append ---
                        # 提取 anchor 和 comparison 的路徑與長度
                        for col in ['anchor', 'comparison']:
                             dur_col = col.replace('anchor', 'anchor_dur').replace('comparison', 'comparison_dur')
                             if dur_col in df.columns:
                                temp_df = df[[col, dur_col]].copy()
                                temp_df.columns = ['wav', 'duration']
                                data_list.append(temp_df)

        if not data_list:
            print("No CSV found!")
            return

        # 一次性合併所有 DataFrame
        data = pd.concat(data_list, ignore_index=True)
        data = data.drop_duplicates(subset=['wav'])
        
        create_npy(data, wav_dir, save_dir, desc="libriphrase")
        return


def preprocess_google(wav_dir = '/home/DB/google_speech_commands',
                      save_dir = '/home/google_speech_embedding/DB/google_speech_commands',
                      target_list = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'],
                      ):

        print(">> Processing Google Speech Commands...")
        data_list = []
        
        for target in target_list:    
                wav_files = [str(x) for x in Path(os.path.join(wav_dir, target)).rglob('*.wav')]
                for wav in wav_files:
                     try:
                        # 這裡只讀取長度，不讀內容
                        import soundfile as sf
                        info = sf.info(wav)
                        dur = info.duration
                        data_list.append({'wav': wav, 'duration': dur})
                     except:
                         pass

        if not data_list:
            print("No GSC files found!")
            return

        # --- [修正] 直接建立 DataFrame ---
        data = pd.DataFrame(data_list)
        create_npy(data, "", save_dir, desc="google")
        return


def preprocess_qualcomm(wav_dir = '/home/DB/qualcomm',
                        save_dir = '/home/google_speech_embedding/DB/qualcomm',
                        target_list=['hey_android', 'hey_snapdragon', 'hi_galaxy', 'hi_lumina'],
                        ):

        print(">> Processing Qualcomm...")
        data_list = []
        
        for target in target_list:    
                wav_files = [str(x) for x in Path(os.path.join(wav_dir, target)).rglob('*.wav')]
                for wav in wav_files:
                     try:
                        import soundfile as sf
                        info = sf.info(wav)
                        dur = info.duration
                        data_list.append({'wav': wav, 'duration': dur})
                     except:
                         pass
        
        if not data_list:
             print("No Qualcomm files found!")
             return

        data = pd.DataFrame(data_list)
        create_npy(data, "", save_dir, desc="qualcomm")
        return


def main():
        base_save_dir='/home/google_speech_embedding/DB/'
        
        #preprocess_libriphrase(save_dir=base_save_dir, train=True)
        #preprocess_libriphrase(save_dir=base_save_dir, train=False)
        preprocess_google(save_dir=base_save_dir)
        #preprocess_qualcomm(save_dir=base_save_dir)


if __name__ == "__main__":
    main()