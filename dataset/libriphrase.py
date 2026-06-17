import math, os, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import Levenshtein
from multiprocessing import Pool
from scipy.io import wavfile
import torch
import torch.nn as nn
import warnings
import lmdb
import pickle

sys.path.append(os.path.dirname(__file__))
try:
    from g2p.g2p_en.g2p import G2p
except ModuleNotFoundError:
    from .g2p.g2p_en.g2p import G2p
try:
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
except AttributeError:
    # NumPy 2.0+ removed VisibleDeprecationWarning
    pass

class LibriPhraseDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 batch_size,
                 fs = 16000,
                 wav_dir='/padawan/LibriPhrase_diffspk_all',
                 gemb_dir=None,
                 noise_dir='/padawan/noise',
                 csv_dir='/padawan/LibriPhrase/data',
                 train_csv = ['train_100h', 'train_360h'],
                 test_csv = ['train_500h',],
                 types='both', # easy, hard
                 features='g2p_embed', # phoneme, g2p_embed, both ...
                 train=True,
                 shuffle=True,
                 pkl=None,
                 edit_dist=False,
                 frame_length=None,
                 hop_length=None,
                 audio_noise=False,
                 # MFA Frame Labels (Aux CE)
                 frame_labels_path=None,  # Path to frame_labels.lmdb
                 ):
        
        # --- LMDB 路徑列表 ---
        self.lmdb_paths = ['/padawan/lmdb_train_100h', '/padawan/lmdb_train_360h', '/padawan/lmdb_train_500h']
        # --- Gemb LMDB 路徑列表 ---
        self.gemb_lmdb_paths = ['/padawan/lmdb_gemb_100h', '/padawan/lmdb_gemb_360h', '/padawan/lmdb_gemb_500h']
        
        phonemes = ["<pad>", ] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                                    'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 
                                    'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 
                                    'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 
                                    'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 
                                    'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 
                                    'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH', 
                                    ' ']
        
        self.p2idx = {p: idx for idx, p in enumerate(phonemes)}
        self.idx2p = {idx: p for idx, p in enumerate(phonemes)}
        
        self.batch_size = batch_size
        self.fs = fs
        self.wav_dir = wav_dir  
        self.gemb_dir = gemb_dir
        self.csv_dir = csv_dir
        self.noise_dir = noise_dir
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.types = types
        self.features = features
        self.train = train
        self.shuffle = shuffle
        self.pkl = pkl
        self.edit_dist = edit_dist
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.audio_noise = audio_noise
        self.nPhoneme = len(phonemes)
        self.g2p = G2p()
        
        self.use_lmdb = False
        self.envs = []
        self.gemb_envs = []
        
        # MFA Frame Labels LMDB (for Aux CE)
        self.frame_labels_path = frame_labels_path
        self.frame_labels_env = None
        
        valid_lmdbs = [p for p in self.lmdb_paths if os.path.isdir(p) and os.path.exists(os.path.join(p, 'data.mdb'))]
        if valid_lmdbs:
            print(f">> [Info] Audio LMDBs detected. Switching to ultra-fast mode.")
            self.use_lmdb = True
            self.lmdb_paths = valid_lmdbs
            
            if self.gemb_dir is not None:
                valid_gemb_lmdbs = [p for p in self.gemb_lmdb_paths if os.path.isdir(p) and os.path.exists(os.path.join(p, 'data.mdb'))]
                if valid_gemb_lmdbs:
                    print(f">> [Info] Gemb LMDBs detected. Using LMDB for embeddings.")
                    self.gemb_lmdb_paths = valid_gemb_lmdbs
        
        # Frame Labels LMDB
        if self.frame_labels_path is not None and os.path.exists(self.frame_labels_path):
            print(f">> [Aux CE] Frame labels LMDB: {self.frame_labels_path}")
        
        self.__prep__()
        self.on_epoch_end()
    
    def _init_lmdb(self):
        if not self.envs:
            for path in self.lmdb_paths:
                try:
                    env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
                    self.envs.append(env)
                except Exception as e:
                    print(f">> [Error] Failed to open Audio LMDB {path}: {e}")

        if self.gemb_dir is not None and not self.gemb_envs and self.use_lmdb:
            for path in self.gemb_lmdb_paths:
                try:
                    env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
                    self.gemb_envs.append(env)
                except Exception as e:
                    print(f">> [Error] Failed to open Gemb LMDB {path}: {e}")

    def __getstate__(self):
        """
        Custom pickle support: close LMDB environments before serialization.
        This is needed for multiprocessing with 'spawn' context.
        """
        state = self.__dict__.copy()
        # Close and remove LMDB environments (cannot be pickled)
        if self.envs:
            for env in self.envs:
                env.close()
        if self.gemb_envs:
            for env in self.gemb_envs:
                env.close()
        state['envs'] = []
        state['gemb_envs'] = []
        state['frame_labels_env'] = None  # LMDB env cannot be pickled
        # Remove g2p object (will be recreated in worker)
        state['g2p'] = None
        return state

    def __setstate__(self, state):
        """
        Custom unpickle support: restore LMDB environments after deserialization.
        LMDB will be lazily reopened on first access in the worker process.
        """
        self.__dict__.update(state)
        # LMDB will be reopened lazily in _init_lmdb() when first accessed
        # Recreate G2p object in worker process
        if self.g2p is None:
            from .g2p.g2p_en.g2p import G2p
            self.g2p = G2p()

    def __prep__(self):
        if self.train:
            print(">> Preparing noise DB")
            noise_list = [str(x) for x in Path(self.noise_dir).rglob('*.wav')]
            
            # --- [修正] 使用 list 收集 + concatenate，並強制轉 float32 ---
            noise_data_list = []
            for noise in noise_list:
                fs, data = wavfile.read(noise)
                # 轉 float32
                data = data.astype(np.float32) / 32768.0
                data = (data / np.max(data)) * 0.5
                noise_data_list.append(data)
            
            if noise_data_list:
                self.noise = np.concatenate(noise_data_list).astype(np.float32)
            else:
                self.noise = np.array([], dtype=np.float32)
            # -------------------------------------------------------
            
        self.data = pd.DataFrame(columns=['wav_label', 'wav', 'text', 'duration', 'label', 'type'])

        if (self.pkl is not None) and (os.path.isfile(self.pkl)):
            print(">> Load dataset from {}".format(self.pkl))
            self.data = pd.read_pickle(self.pkl)
        else:
            for db in self.train_csv if self.train else self.test_csv:
                csv_list = [str(x) for x in Path(self.csv_dir).rglob('*' + db + '*word*')]
                if not csv_list:
                     csv_list = [str(x) for x in Path(self.csv_dir).rglob('*' + db + '*.csv')]

                for n_word in csv_list:
                    print(">> processing : {} ".format(n_word))
                    df = pd.read_csv(n_word)
                    # Split train dataset to match & unmatch case
                    anc_pos = df[['anchor_text', 'anchor', 'anchor_text', 'anchor_dur']]
                    anc_neg = df[['anchor_text', 'anchor', 'comparison_text', 'anchor_dur', 'target', 'type']]
                    com_pos = df[['comparison_text', 'comparison', 'comparison_text', 'comparison_dur']]
                    com_neg = df[['comparison_text', 'comparison', 'anchor_text', 'comparison_dur', 'target', 'type']]
                    anc_pos.columns = ['wav_label', 'anchor', 'anchor_text', 'anchor_dur']
                    com_pos.columns = ['wav_label', 'comparison', 'comparison_text', 'comparison_dur']
                    anc_pos['label'] = 1
                    anc_pos['type'] = df['type']
                    com_pos['label'] = 1
                    com_pos['type'] = df['type']
                    # Concat
                    self.data = pd.concat([self.data, anc_pos.rename(columns={y: x for x, y in zip(self.data.columns, anc_pos.columns)})], ignore_index=True)
                    self.data = pd.concat([self.data, anc_neg.rename(columns={y: x for x, y in zip(self.data.columns, anc_neg.columns)})], ignore_index=True)
                    self.data = pd.concat([self.data, com_pos.rename(columns={y: x for x, y in zip(self.data.columns, com_pos.columns)})], ignore_index=True)
                    self.data = pd.concat([self.data, com_neg.rename(columns={y: x for x, y in zip(self.data.columns, com_neg.columns)})], ignore_index=True)
            
            if self.use_lmdb:
                self.data['wav'] = self.data['wav'].astype(str)
            else:
                self.data['wav'] = self.data['wav'].apply(lambda x: os.path.join(self.wav_dir, x))
            
            original_len = len(self.data)
            self.data = self.data.dropna(subset=['text', 'wav_label'])
            self.data['text'] = self.data['text'].astype(str)
            self.data['wav_label'] = self.data['wav_label'].astype(str)
            if len(self.data) < original_len:
                print(f">> [Warning] Dropped {original_len - len(self.data)} rows.")

            print(">> Convert word to phoneme")
            self.data['phoneme'] = self.data['text'].apply(lambda x: self.g2p(re.sub(r"[^a-zA-Z0-9]+", ' ', x)))
            print(">> Convert speech word to phoneme")
            self.data['wav_phoneme'] = self.data['wav_label'].apply(lambda x: self.g2p(re.sub(r"[^a-zA-Z0-9]+", ' ', x)))
            print(">> Convert phoneme to index")
            self.data['pIndex'] = self.data['phoneme'].apply(lambda x: [self.p2idx[t] for t in x])
            print(">> Convert speech phoneme to index")
            self.data['wav_pIndex'] = self.data['wav_phoneme'].apply(lambda x: [self.p2idx[t] for t in x])
            print(">> Compute phoneme embedding")
            self.data['g2p_embed'] = self.data['text'].apply(lambda x: self.g2p.embedding(x))
            print(">> Calucate Edit distance ratio")
            self.data['dist'] = self.data.apply(lambda x: Levenshtein.ratio(re.sub(r"[^a-zA-Z0-9]+", ' ', x['wav_label']), re.sub(r"[^a-zA-Z0-9]+", ' ', x['text'])), axis=1)

            if (self.pkl is not None) and (not os.path.isfile(self.pkl)):
                self.data.to_pickle(self.pkl)
        
        if self.types == 'both':
            pass
        elif self.types == 'easy':
            self.data = self.data.loc[self.data['type'] == 'diffspk_easyneg']
        elif self.types == 'hard':
            self.data = self.data.loc[self.data['type'] == 'diffspk_hardneg']

        self.data = self.data.sort_values(by='duration').reset_index(drop=True)
        self.wav_list = self.data['wav'].values
        self.idx_list = self.data['pIndex'].values
        self.sIdx_list = self.data['wav_pIndex'].values
        self.emb_list = self.data['g2p_embed'].values
        self.lab_list = self.data['label'].values
        if self.edit_dist:
            self.dist_list = self.data['dist'].values
        
        self.len = len(self.data)
        self.maxlen_t = int((int(self.data['text'].apply(lambda x: len(x)).max() / 10) + 1) * 10)
        self.maxlen_a = int((int(self.data['duration'].values[-1] / 0.5) + 1 ) * self.fs / 2)
        self.maxlen_l = int((int(self.data['wav_label'].apply(lambda x: len(x)).max() / 10) + 1) * 10)
        
        # === Memory Optimization: Extract arrays before DataFrame deletion ===
        # Extract text list (needed for potential on-the-fly g2p in future)
        self.text_list = self.data['text'].values  # ~20MB
        
        # Extract type list (needed for easy/hard validation split)
        self.type_list = self.data['type'].values  # ~1MB
        
        # NOTE: DataFrame deletion moved to subclass (PersonalizedLibriPhraseDataset)
        # because _prepare_speaker_info() needs self.data to extract speaker IDs.
        # After speaker info is prepared, subclass will delete self.data
                            
    def __len__(self):
        return self.len

    def _load_wav(self, wav_path):
        if self.use_lmdb:
            if not self.envs:
                self._init_lmdb()
            
            # 移除絕對路徑前綴 (既有邏輯)
            prefix_to_remove = '/padawan/LibriPhrase_diffspk_all/'
            if wav_path.startswith(prefix_to_remove):
                wav_path = wav_path.replace(prefix_to_remove, '')
                
            for env in self.envs:
                with env.begin(write=False) as txn:
                    # 1. 嘗試原始 Key
                    try:
                        key = wav_path.encode('ascii')
                    except UnicodeEncodeError:
                        key = wav_path.encode('utf-8')
                    
                    byte_data = txn.get(key)
                    
                    # 2. 嘗試 .wav -> .flac
                    if byte_data is None and wav_path.endswith('.wav'):
                        key_flac = wav_path.replace('.wav', '.flac').encode('ascii')
                        byte_data = txn.get(key_flac)

                    # 3. [新增] 嘗試修復重複目錄 (例如 train-clean-360/train-clean-360 -> train-clean-360)
                    if byte_data is None:
                        parts = wav_path.split(os.sep)
                        if len(parts) > 1 and parts[0] == parts[1]:
                            key_dedup = os.path.join(*parts[1:]).encode('ascii')
                            byte_data = txn.get(key_dedup)
                            # 如果這樣還找不到，試試 dedup + flac
                            if byte_data is None and wav_path.endswith('.wav'):
                                key_dedup_flac = os.path.join(*parts[1:]).replace('.wav', '.flac').encode('ascii')
                                byte_data = txn.get(key_dedup_flac)

                    if byte_data is not None:
                        audio_int16 = pickle.loads(byte_data)
                        return audio_int16.astype(np.float32) / 32768.0
            
            # --- [新增] 終極防崩潰機制 ---
            # 如果跑遍所有 LMDB 都找不到，印個警告並回傳全零靜音
            # print(f"[Warning] Key '{wav_path}' missing in LMDB. Returning silence.")
            return np.zeros(self.maxlen_a, dtype=np.float32)
            
        else:
            # Raw mode fallback
            try:
                return np.array(wavfile.read(wav_path)[1]).astype(np.float32) / 32768.0
            except Exception:
                return np.zeros(self.maxlen_a, dtype=np.float32)
    
    def _mixing_snr(self, clean, snr=[5, 15]):
        def _cal_adjusted_rms(clean_rms, snr):
            a = float(snr) / 20
            noise_rms = clean_rms / (10**a) 
            return noise_rms

        def _cal_rms(amp):
            return np.sqrt(np.mean(np.square(amp), axis=-1))
        
        start = np.random.randint(0, len(self.noise)-len(clean))
        divided_noise = self.noise[start: start + len(clean)]
        
        clean_rms = _cal_rms(clean)
        noise_rms = _cal_rms(divided_noise)
        adj_noise_rms = _cal_adjusted_rms(clean_rms, np.random.randint(snr[0], snr[1]))
        
        adj_noise_amp = divided_noise * (adj_noise_rms / (noise_rms + 1e-7)) 
        noisy = clean + adj_noise_amp
        
        if np.max(noisy) > 1:
            noisy = noisy / np.max(noisy)
        
        return noisy
    
    def _load_frame_labels(self, wav_path):
        """
        Load MFA frame-level phoneme labels from LMDB.
        
        Args:
            wav_path: Path to wav file (used to derive the key)
            
        Returns:
            np.array of phoneme IDs, or None if not found
        """
        if self.frame_labels_path is None:
            return None
        
        # Initialize LMDB if not already done
        if self.frame_labels_env is None:
            if os.path.exists(self.frame_labels_path):
                self.frame_labels_env = lmdb.open(
                    self.frame_labels_path, 
                    readonly=True, 
                    lock=False, 
                    readahead=False, 
                    meminit=False
                )
            else:
                return None
        
        # Extract file ID from wav path
        # wav_path format: "train-clean-360/speaker/chapter/speaker-chapter-utt.wav"
        # We need to find matching key in LMDB (format: "speaker-chapter-utt_Nword_idx")
        basename = os.path.basename(wav_path)
        file_stem = os.path.splitext(basename)[0]  # e.g., "1121-132777-0001"
        
        with self.frame_labels_env.begin(write=False) as txn:
            # Try to find a key that starts with this file_stem
            # LMDB keys are like "1121-132777-0001_2word_34074"
            cursor = txn.cursor()
            prefix = file_stem.encode('utf-8')
            
            # Look for prefix match
            if cursor.set_range(prefix):
                key, value = cursor.item()
                if key.startswith(prefix):
                    return pickle.loads(value)
        
        return None
    
    def __getitem__(self, idx):
        i = self.indices[idx]
        x_np = self._load_wav(self.wav_list[i]) 

        if self.features == 'both':
            p = torch.Tensor(self.idx_list[i]).to(torch.int32)
            e = torch.Tensor(self.emb_list[i]).to(torch.float32)
        else:
            if self.features == 'phoneme':
                y = torch.Tensor(self.idx_list[i]).to(torch.int32)
            elif self.features == 'g2p_embed':
                y = torch.Tensor(self.emb_list[i]).to(torch.float32)
        
        z = torch.Tensor([self.lab_list[i]]).to(torch.float32)
        l = torch.Tensor(self.sIdx_list[i]).to(torch.int32)
        t = torch.Tensor(self.idx_list[i]).to(torch.int32)
        if self.edit_dist:
            d = torch.Tensor([self.dist_list[i]]).to(torch.float32)

        if self.train and self.audio_noise:
            x_noisy_np = self._mixing_snr(x_np)
        else:
            x_noisy_np = x_np
        
        # --- [修正] 確保這裡是 Float32 ---
        x = torch.from_numpy(x_np).float()
        x_noisy = torch.from_numpy(x_noisy_np).float() if self.train else None
        # ------------------------------

        if self.gemb_dir is not None:
            gemb_np = self._load_gemb(self.wav_list[i])
            gemb = torch.from_numpy(gemb_np).to(torch.float32)
        else:
            gemb = None
        
        # Load MFA frame labels (for Aux CE)
        frame_labels = None
        if self.frame_labels_path is not None:
            fl_np = self._load_frame_labels(self.wav_list[i])
            if fl_np is not None:
                frame_labels = torch.from_numpy(fl_np).to(torch.int64)
            
        if self.train:
            if self.features == 'both':
                return {"x": x, "x_noisy": x_noisy, "gemb": gemb, "y": None, "p": p, "e": e, "z": z, "l": l, "t": t, "d": None, "frame_labels": frame_labels}
            else:
                return {"x": x, "x_noisy": x_noisy, "gemb": gemb, "y": y, "p": None, "e": None, "z": z, "l": l, "t": t, "d": None, "frame_labels": frame_labels}
        else:
            if self.features == 'both':
                if self.edit_dist:
                    return {"x": x, "x_noisy": None, "gemb": gemb, "y": None, "p": p, "e": e, "z": z, "l": None, "t": None, "d": d, "frame_labels": frame_labels}
                else:
                    return {"x": x, "x_noisy": None, "gemb": gemb, "y": None, "p": p, "e": e, "z": z, "l": None, "t": None, "d": None, "frame_labels": frame_labels}
            else:
                if self.edit_dist:
                    return {"x": x, "x_noisy": None, "gemb": gemb, "y": y, "p": None, "e": None, "z": z, "l": None, "t": None, "d": d, "frame_labels": frame_labels}
                else:
                    return {"x": x, "x_noisy": None, "gemb": gemb, "y": y, "p": None, "e": None, "z": z, "l": None, "t": None, "d": None, "frame_labels": frame_labels}

    def on_epoch_end(self):
        self.indices = np.arange(self.len)
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def pad_sequence(self, data, max_len):
        pad_list = [0 for _ in range(data[0].dim()*2)]
        pad_list[-1] = max_len - data[0].shape[0]
        data[0] = torch.nn.functional.pad(data[0], tuple(pad_list))
        return torch.nn.utils.rnn.pad_sequence(data, batch_first=True)

    def collate(self, batch):
        batch_dict = {
            "x": None,          "x_len": None,
            "x_noisy": None,    "x_noisy_len": None,
            "gemb": None,       "gemb_len": None, 
            "y": None,          "y_len": None,
            "p": None,          "p_len": None,
            "e": None,          "e_len": None,
            "z": None,          "z_len": None,
            "l": None,          "l_len": None,
            "t": None,          "t_len": None,
            "d": None,          "d_len": None,
            "frame_labels": None,  "frame_labels_len": None,  # MFA frame labels
            }
        
        device = batch[0]["x"].device
        batch_dict["x"] = self.pad_sequence([b["x"] for b in batch], self.maxlen_a)
        batch_dict["z"] = torch.nn.utils.rnn.pad_sequence([b["z"] for b in batch], batch_first=True)
        batch_dict["x_len"] = torch.Tensor([b["x"].shape[0] for b in batch]).to(dtype=torch.int32, device=device)
        
        if self.features == 'both':
            batch_dict["p"] = self.pad_sequence([b["p"] for b in batch], self.maxlen_t)
            batch_dict["e"] = self.pad_sequence([b["e"] for b in batch], self.maxlen_t)
            batch_dict["p_len"] = torch.Tensor([b["p"].shape[0] for b in batch]).to(dtype=torch.int32, device=device)
            batch_dict["e_len"] = torch.Tensor([b["e"].shape[0] for b in batch]).to(dtype=torch.int32, device=device)
        else:
            batch_dict["y"] = self.pad_sequence([b["y"] for b in batch], self.maxlen_t)
            batch_dict["y_len"] = torch.Tensor([b["y"].shape[0] for b in batch]).to(dtype=torch.int32, device=device)
        
        if self.train:
            batch_dict["x_noisy"] = self.pad_sequence([b["x_noisy"] for b in batch], self.maxlen_a)
            batch_dict["l"] = self.pad_sequence([b["l"] for b in batch], self.maxlen_l)
            batch_dict["t"] = self.pad_sequence([b["t"] for b in batch], self.maxlen_t)
            batch_dict["l_len"] = torch.Tensor([b["l"].shape[0] for b in batch]).to(dtype=torch.int32, device=device)
            batch_dict["t_len"] = torch.Tensor([b["t"].shape[0] for b in batch]).to(dtype=torch.int32, device=device)
        
        if self.gemb_dir is not None:
            batch_dict["gemb"] = self.pad_sequence([b["gemb"] for b in batch], int(int((self.maxlen_a - self.frame_length)/self.hop_length + 1)/8))
            batch_dict["gemb_len"] = torch.Tensor([int(int((b["x"].shape[0] - self.frame_length)/self.hop_length + 1)/8) for b in batch]).to(dtype=torch.int32, device=device)
        
        elif self.edit_dist:
            batch_dict["d"] = torch.nn.utils.rnn.pad_sequence([b["d"] for b in batch], batch_first=True)
            batch_dict["d_len"] = torch.Tensor([b["d"].shape[0] for b in batch]).to(dtype=torch.int32, device=device)

        # MFA frame labels (for Aux CE)
        if self.frame_labels_path is not None:
            # Filter samples with valid frame_labels
            valid_fl = [b["frame_labels"] for b in batch if b["frame_labels"] is not None]
            if len(valid_fl) == len(batch):
                # All samples have frame labels - pad with PAD=0
                batch_dict["frame_labels"] = torch.nn.utils.rnn.pad_sequence(
                    valid_fl, batch_first=True, padding_value=0
                )
                batch_dict["frame_labels_len"] = torch.Tensor(
                    [fl.shape[0] for fl in valid_fl]
                ).to(dtype=torch.int32, device=device)
            # If any sample is missing frame labels, leave as None

        return batch_dict
    
    def _load_gemb(self, wav_path):
        # 1. 優先嘗試 LMDB
        if self.use_lmdb and self.gemb_envs:
            if not self.gemb_envs:
                 self._init_lmdb()
            
            prefix_to_remove = '/padawan/LibriPhrase_diffspk_all/'
            key_path = wav_path
            if wav_path.startswith(prefix_to_remove):
                key_path = wav_path.replace(prefix_to_remove, '')
            
            for env in self.gemb_envs:
                with env.begin(write=False) as txn:
                     try:
                        key = key_path.encode('ascii')
                     except UnicodeEncodeError:
                        key = key_path.encode('utf-8')
                     
                     byte_data = txn.get(key)
                     if byte_data is not None:
                         data = pickle.loads(byte_data)
                         if data.ndim == 3 and data.shape[0] == 1:
                             data = data[0]
                         return data.astype(np.float32)
        
        # 2. Fallback: 讀取 .npy 檔案
        filename = os.path.basename(wav_path)
        filename_no_ext = os.path.splitext(filename)[0]
        npy_filename = filename_no_ext + '.npy'
        
        prefix_to_remove = '/padawan/LibriPhrase_diffspk_all/'
        rel_path = wav_path
        if wav_path.startswith(prefix_to_remove):
            rel_path = wav_path.replace(prefix_to_remove, '')
            
        parts = rel_path.split(os.sep)
        if len(parts) > 1:
            dirpath = os.path.dirname(rel_path)
            candidate1 = os.path.join(self.gemb_dir, dirpath, npy_filename)
            if os.path.exists(candidate1):
                return np.load(candidate1)[0].astype(np.float32)
            
            dirpath_short = os.path.join(*parts[1:-1])
            candidate2 = os.path.join(self.gemb_dir, dirpath_short, npy_filename)
            if os.path.exists(candidate2):
                return np.load(candidate2)[0].astype(np.float32)
        
        raise FileNotFoundError(f"Gemb NPY not found for {wav_path}. Searched in {self.gemb_dir}")