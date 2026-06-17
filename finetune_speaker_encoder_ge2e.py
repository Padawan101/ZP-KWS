"""
Speaker Encoder Finetuning with GE2E Loss

This script finetunes EfficientTDNN on LibriPhrase dataset using the
Generalized End-to-End (GE2E) loss for improved speaker verification
performance on short utterances.

GE2E Loss: https://arxiv.org/abs/1710.10467
Key difference from contrastive/BCE approach:
- Requires batches of N speakers × M utterances per speaker
- Learns speaker centroids and optimizes similarity matrix

Usage:
    python speaker/finetune_speaker_encoder_ge2e.py \
        --train_pkl /padawan/train_combined.pkl \
        --val_pkl /padawan/test_500h.pkl \
        --epochs 30 \
        --n_speakers 16 \
        --n_utterances 10 \
        --output_dir ./checkpoints/speaker_encoder_ge2e

Author: Claude Code
Date: 2026-01-12
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm
import pickle
import pandas as pd
from sklearn.metrics import roc_curve, auc
from pathlib import Path
import lmdb
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# 0. Audio Augmentation
# ============================================================

class AudioAugmentor:
    """
    Audio augmentation for speaker encoder training.
    Supports:
    - Additive noise (various types: babble, office, traffic, etc.)
    - Reverberation (optional, if RIR available)
    
    Compatible with noise corpus at /padawan/noise/noise_train
    """
    
    def __init__(
        self,
        noise_dir: str = None,
        rir_dir: str = None,
        sample_rate: int = 16000,
        snr_range: tuple = (5, 20),  # SNR in dB
        seed: int = 42,
    ):
        """
        Args:
            noise_dir: Path to noise wav files (e.g., /padawan/noise/noise_train)
            rir_dir: Optional path to RIR files for reverberation
            sample_rate: Audio sample rate
            snr_range: (min_snr, max_snr) in dB for additive noise
            seed: Random seed
        """
        self.sample_rate = sample_rate
        self.snr_range = snr_range
        self.rng = np.random.RandomState(seed)
        
        # Load noise files
        self.noise_files = []
        if noise_dir and os.path.isdir(noise_dir):
            import glob
            self.noise_files = glob.glob(os.path.join(noise_dir, '*.wav'))
            print(f">> AudioAugmentor: Loaded {len(self.noise_files)} noise files from {noise_dir}")
        
        # Categorize noise files
        self.noise_categories = {
            'babble': [],    # Babble, NeighborSpeaking
            'ambient': [],   # Office, Kitchen, Car, etc.
            'mechanical': [], # AirConditioner, VacuumCleaner, etc.
        }
        for nf in self.noise_files:
            basename = os.path.basename(nf).lower()
            if 'babble' in basename or 'neighbor' in basename or 'speaking' in basename:
                self.noise_categories['babble'].append(nf)
            elif 'air' in basename or 'vacuum' in basename or 'washer' in basename or 'copy' in basename:
                self.noise_categories['mechanical'].append(nf)
            else:
                self.noise_categories['ambient'].append(nf)
        
        # Load RIR files
        self.rir_files = []
        if rir_dir and os.path.isdir(rir_dir):
            import glob
            self.rir_files = glob.glob(os.path.join(rir_dir, '**/*.wav'), recursive=True)
            print(f">> AudioAugmentor: Loaded {len(self.rir_files)} RIR files from {rir_dir}")
        
        # Pre-load noise files to memory for speed
        self.noise_cache = {}
        self._preload_noise(max_files=50)  # Limit to 50 for memory
    
    def _preload_noise(self, max_files: int = 50):
        """Preload noise files to memory."""
        from scipy.io import wavfile
        for nf in self.noise_files[:max_files]:
            try:
                _, data = wavfile.read(nf)
                self.noise_cache[nf] = data.astype(np.float32) / 32768.0
            except Exception:
                pass
    
    def _load_noise(self, noise_path: str, target_len: int) -> np.ndarray:
        """Load and adjust noise length."""
        if noise_path in self.noise_cache:
            noise = self.noise_cache[noise_path]
        else:
            from scipy.io import wavfile
            try:
                _, noise = wavfile.read(noise_path)
                noise = noise.astype(np.float32) / 32768.0
            except Exception:
                return np.zeros(target_len, dtype=np.float32)
        
        # Adjust length
        if len(noise) < target_len:
            # Repeat to fill
            repeats = int(np.ceil(target_len / len(noise)))
            noise = np.tile(noise, repeats)[:target_len]
        elif len(noise) > target_len:
            # Random crop
            start = self.rng.randint(0, len(noise) - target_len)
            noise = noise[start:start + target_len]
        
        return noise
    
    def add_noise(self, audio: np.ndarray, category: str = None) -> np.ndarray:
        """
        Add noise to audio at random SNR.
        
        Args:
            audio: Clean audio [T]
            category: Optional noise category ('babble', 'ambient', 'mechanical')
        
        Returns:
            Noisy audio [T]
        """
        if len(self.noise_files) == 0:
            return audio
        
        # Select noise file
        if category and category in self.noise_categories and self.noise_categories[category]:
            noise_file = self.rng.choice(self.noise_categories[category])
        else:
            noise_file = self.rng.choice(self.noise_files)
        
        noise = self._load_noise(noise_file, len(audio))
        
        # Calculate SNR
        snr_db = self.rng.uniform(self.snr_range[0], self.snr_range[1])
        
        # Calculate scaling factor
        audio_power = np.mean(audio ** 2) + 1e-8
        noise_power = np.mean(noise ** 2) + 1e-8
        
        scale = np.sqrt(audio_power / (noise_power * (10 ** (snr_db / 10))))
        
        return audio + scale * noise
    
    def reverberate(self, audio: np.ndarray) -> np.ndarray:
        """Apply reverberation using RIR."""
        if len(self.rir_files) == 0:
            return audio
        
        from scipy.io import wavfile
        from scipy import signal
        
        rir_file = self.rng.choice(self.rir_files)
        try:
            _, rir = wavfile.read(rir_file)
            rir = rir.astype(np.float32) / (np.sqrt(np.sum(rir ** 2)) + 1e-8)
            
            # Convolve
            reverbed = signal.fftconvolve(audio, rir, mode='full')[:len(audio)]
            return reverbed.astype(np.float32)
        except Exception:
            return audio
    
    def augment(self, audio: np.ndarray, p_noise: float = 0.7, p_reverb: float = 0.3) -> np.ndarray:
        """
        Apply random augmentation.
        
        Args:
            audio: Clean audio [T]
            p_noise: Probability of adding noise
            p_reverb: Probability of adding reverb
        
        Returns:
            Augmented audio [T]
        """
        if self.rng.random() < p_noise and len(self.noise_files) > 0:
            # Choose noise category
            categories = ['babble', 'ambient', 'mechanical', None]
            category = self.rng.choice(categories)
            audio = self.add_noise(audio, category=category)
        
        if self.rng.random() < p_reverb and len(self.rir_files) > 0:
            audio = self.reverberate(audio)
        
        # Clip to prevent overflow
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio


# ============================================================
# 1. Dataset: Speaker-Grouped Batch for GE2E
# ============================================================

class GE2ESpeakerDataset(Dataset):
    """
    Dataset for GE2E training.
    Returns individual utterances, grouping is done by the sampler.
    """
    
    def __init__(
        self,
        pkl_path: str,
        min_utterances_per_speaker: int = 10,
        min_audio_length: int = 3200,  # 0.2 seconds at 16kHz
        max_audio_length: int = 80000,  # 5 seconds at 16kHz
        augmentor: AudioAugmentor = None,
        augment_prob: float = 0.0,  # Probability of augmentation per sample
    ):
        """
        Args:
            pkl_path: Path to LibriPhrase pickle file
            min_utterances_per_speaker: Minimum utterances required per speaker
            min_audio_length: Minimum audio samples
            max_audio_length: Maximum audio samples (truncate longer)
            augmentor: AudioAugmentor instance for noise augmentation
            augment_prob: Probability of applying augmentation (0.0 = off, 1.0 = always)
        """
        self.min_audio_length = min_audio_length
        self.max_audio_length = max_audio_length
        self.rng = np.random.RandomState(42)
        self.augmentor = augmentor
        self.augment_prob = augment_prob
        
        # LMDB paths (same as libriphrase.py)
        self.lmdb_paths = [
            '/padawan/lmdb_train_100h',
            '/padawan/lmdb_train_360h',
            '/padawan/lmdb_train_500h'
        ]
        self.envs = []
        self.use_lmdb = any(
            os.path.isdir(p) and os.path.exists(os.path.join(p, 'data.mdb'))
            for p in self.lmdb_paths
        )
        
        # Load data
        print(f">> Loading data from {pkl_path}...")
        self.data = pd.read_pickle(pkl_path)
        print(f">> Loaded {len(self.data)} samples")
        
        # Extract wav paths
        self.wav_list = self.data['wav'].values
        
        # Extract speaker IDs and build mapping
        print(">> Extracting speaker IDs...")
        self.speaker_ids = [self._extract_speaker_id(w) for w in self.wav_list]
        
        # Build speaker -> indices mapping
        self.speaker_to_indices = defaultdict(list)
        for idx, spk in enumerate(self.speaker_ids):
            self.speaker_to_indices[spk].append(idx)
        
        # Filter speakers with enough utterances
        self.valid_speakers = [
            spk for spk, indices in self.speaker_to_indices.items()
            if len(indices) >= min_utterances_per_speaker
        ]
        
        # Build flat index list with speaker info for each valid sample
        self.samples = []  # (global_idx, speaker_id)
        for spk in self.valid_speakers:
            for idx in self.speaker_to_indices[spk]:
                self.samples.append((idx, spk))
        
        print(f">> {len(self.valid_speakers)} speakers with >= {min_utterances_per_speaker} utterances")
        print(f">> Total {len(self.samples)} valid samples")
    
    def _extract_speaker_id(self, wav_path: str) -> str:
        """Extract speaker ID from LibriSpeech-style path."""
        filename = os.path.basename(wav_path)
        filename = filename.replace('.wav', '').replace('.flac', '')
        parts = filename.split('-')
        return parts[0] if parts else "unknown"
    
    def _init_lmdb(self):
        """Lazily initialize LMDB connections."""
        if not self.envs:
            valid_paths = [
                p for p in self.lmdb_paths
                if os.path.isdir(p) and os.path.exists(os.path.join(p, 'data.mdb'))
            ]
            for path in valid_paths:
                try:
                    env = lmdb.open(
                        path, readonly=True, lock=False,
                        readahead=False, meminit=False
                    )
                    self.envs.append(env)
                except Exception as e:
                    print(f">> [Warning] Failed to open LMDB {path}: {e}")
    
    def _load_audio(self, wav_path: str) -> np.ndarray:
        """Load audio from LMDB or filesystem."""
        if self.use_lmdb:
            if not self.envs:
                self._init_lmdb()
            
            prefix = '/padawan/LibriPhrase_diffspk_all/'
            key_path = wav_path
            if wav_path.startswith(prefix):
                key_path = wav_path.replace(prefix, '')
            
            for env in self.envs:
                with env.begin(write=False) as txn:
                    try:
                        key = key_path.encode('ascii')
                    except UnicodeEncodeError:
                        key = key_path.encode('utf-8')
                    
                    byte_data = txn.get(key)
                    
                    if byte_data is None and key_path.endswith('.wav'):
                        key_flac = key_path.replace('.wav', '.flac').encode('ascii')
                        byte_data = txn.get(key_flac)
                    
                    if byte_data is None:
                        parts = key_path.split(os.sep)
                        if len(parts) > 1 and parts[0] == parts[1]:
                            key_dedup = os.path.join(*parts[1:]).encode('ascii')
                            byte_data = txn.get(key_dedup)
                    
                    if byte_data is not None:
                        audio_int16 = pickle.loads(byte_data)
                        return audio_int16.astype(np.float32) / 32768.0
            
            return np.zeros(self.min_audio_length, dtype=np.float32)
        else:
            from scipy.io import wavfile
            try:
                _, data = wavfile.read(wav_path)
                return data.astype(np.float32) / 32768.0
            except Exception:
                return np.zeros(self.min_audio_length, dtype=np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        global_idx, speaker_id = self.samples[idx]
        
        audio = self._load_audio(self.wav_list[global_idx])
        
        # Pad if too short
        if len(audio) < self.min_audio_length:
            audio = np.pad(audio, (0, self.min_audio_length - len(audio)), mode='wrap')
        
        # Random crop if too long
        if len(audio) > self.max_audio_length:
            start = self.rng.randint(0, len(audio) - self.max_audio_length)
            audio = audio[start:start + self.max_audio_length]
        
        # Apply augmentation (training only)
        if self.augmentor is not None and self.augment_prob > 0:
            if self.rng.random() < self.augment_prob:
                audio = self.augmentor.augment(audio)
        
        return {
            'audio': torch.from_numpy(audio).float(),
            'speaker_id': speaker_id,
            'sample_idx': global_idx,
        }
    
    def __getstate__(self):
        state = self.__dict__.copy()
        if self.envs:
            for env in self.envs:
                try:
                    env.close()
                except:
                    pass
        state['envs'] = []
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)


class GE2EBatchSampler(Sampler):
    """
    Sampler that creates batches of N speakers × M utterances.
    
    Each batch contains exactly n_speakers different speakers,
    with n_utterances per speaker.
    """
    
    def __init__(
        self,
        dataset: GE2ESpeakerDataset,
        n_speakers: int,
        n_utterances: int,
        num_batches: int = 1000,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.n_speakers = n_speakers
        self.n_utterances = n_utterances
        self.num_batches = num_batches
        self.rng = np.random.RandomState(seed)
        
        # Build speaker -> sample indices mapping
        self.speaker_to_sample_indices = defaultdict(list)
        for sample_idx, (_, speaker_id) in enumerate(dataset.samples):
            self.speaker_to_sample_indices[speaker_id].append(sample_idx)
        
        # Filter speakers with enough utterances
        self.valid_speakers = [
            spk for spk, indices in self.speaker_to_sample_indices.items()
            if len(indices) >= n_utterances
        ]
        
        if len(self.valid_speakers) < n_speakers:
            raise ValueError(
                f"Not enough speakers with >= {n_utterances} utterances. "
                f"Found {len(self.valid_speakers)}, need {n_speakers}"
            )
        
        print(f">> GE2EBatchSampler: {len(self.valid_speakers)} valid speakers")
    
    def __iter__(self):
        for _ in range(self.num_batches):
            # Sample n_speakers different speakers
            speakers = self.rng.choice(
                self.valid_speakers, self.n_speakers, replace=False
            )
            
            batch = []
            for spk in speakers:
                # Sample n_utterances for this speaker
                indices = self.speaker_to_sample_indices[spk]
                selected = self.rng.choice(
                    indices, self.n_utterances, replace=False
                )
                batch.extend(selected.tolist())
            
            yield batch
    
    def __len__(self):
        return self.num_batches


def ge2e_collate_fn(batch):
    """Collate function with padding for variable-length audio."""
    max_len = max(b['audio'].shape[0] for b in batch)
    
    audio_batch = []
    speaker_ids = []
    
    for b in batch:
        pad_len = max_len - b['audio'].shape[0]
        if pad_len > 0:
            padded = F.pad(b['audio'], (0, pad_len))
        else:
            padded = b['audio']
        audio_batch.append(padded)
        speaker_ids.append(b['speaker_id'])
    
    return {
        'audio': torch.stack(audio_batch),
        'speaker_ids': speaker_ids,
    }


# ============================================================
# 2. Validation Dataset (Pair-based, same as original)
# ============================================================

class ValidationPairDataset(Dataset):
    """Pair-based dataset for validation (EER computation)."""
    
    def __init__(
        self,
        pkl_path: str,
        num_pairs: int = 20000,
        seed: int = 42,
        min_audio_length: int = 3200,
        max_audio_length: int = 80000,
    ):
        self.num_pairs = num_pairs
        self.rng = np.random.RandomState(seed)
        self.min_audio_length = min_audio_length
        self.max_audio_length = max_audio_length
        
        # LMDB setup
        self.lmdb_paths = [
            '/padawan/lmdb_train_100h',
            '/padawan/lmdb_train_360h',
            '/padawan/lmdb_train_500h'
        ]
        self.envs = []
        self.use_lmdb = any(
            os.path.isdir(p) and os.path.exists(os.path.join(p, 'data.mdb'))
            for p in self.lmdb_paths
        )
        
        # Load data
        self.data = pd.read_pickle(pkl_path)
        self.wav_list = self.data['wav'].values
        
        # Extract speaker IDs
        self.speaker_ids = [self._extract_speaker_id(w) for w in self.wav_list]
        
        # Build speaker mapping
        self.speaker_to_indices = defaultdict(list)
        for idx, spk in enumerate(self.speaker_ids):
            self.speaker_to_indices[spk].append(idx)
        
        self.speakers = list(self.speaker_to_indices.keys())
        self.valid_speakers = [
            spk for spk in self.speakers
            if len(self.speaker_to_indices[spk]) >= 2
        ]
        
        # Generate pairs
        self.pairs = self._generate_pairs()
    
    def _extract_speaker_id(self, wav_path: str) -> str:
        filename = os.path.basename(wav_path)
        filename = filename.replace('.wav', '').replace('.flac', '')
        parts = filename.split('-')
        return parts[0] if parts else "unknown"
    
    def _generate_pairs(self):
        pairs = []
        for _ in range(self.num_pairs):
            if self.rng.random() < 0.5:
                # Positive pair
                spk = self.rng.choice(self.valid_speakers)
                indices = self.speaker_to_indices[spk]
                if len(indices) >= 2:
                    idx1, idx2 = self.rng.choice(len(indices), 2, replace=False)
                    pairs.append((indices[idx1], indices[idx2], 1))
            else:
                # Negative pair
                spk1, spk2 = self.rng.choice(self.speakers, 2, replace=False)
                idx1 = self.rng.choice(self.speaker_to_indices[spk1])
                idx2 = self.rng.choice(self.speaker_to_indices[spk2])
                pairs.append((idx1, idx2, 0))
        return pairs
    
    def _init_lmdb(self):
        if not self.envs:
            valid_paths = [
                p for p in self.lmdb_paths
                if os.path.isdir(p) and os.path.exists(os.path.join(p, 'data.mdb'))
            ]
            for path in valid_paths:
                try:
                    env = lmdb.open(path, readonly=True, lock=False,
                                    readahead=False, meminit=False)
                    self.envs.append(env)
                except Exception:
                    pass
    
    def _load_audio(self, wav_path: str) -> np.ndarray:
        if self.use_lmdb:
            if not self.envs:
                self._init_lmdb()
            
            prefix = '/padawan/LibriPhrase_diffspk_all/'
            key_path = wav_path
            if wav_path.startswith(prefix):
                key_path = wav_path.replace(prefix, '')
            
            for env in self.envs:
                with env.begin(write=False) as txn:
                    try:
                        key = key_path.encode('ascii')
                    except UnicodeEncodeError:
                        key = key_path.encode('utf-8')
                    
                    byte_data = txn.get(key)
                    
                    if byte_data is None and key_path.endswith('.wav'):
                        key_flac = key_path.replace('.wav', '.flac').encode('ascii')
                        byte_data = txn.get(key_flac)
                    
                    if byte_data is None:
                        parts = key_path.split(os.sep)
                        if len(parts) > 1 and parts[0] == parts[1]:
                            key_dedup = os.path.join(*parts[1:]).encode('ascii')
                            byte_data = txn.get(key_dedup)
                    
                    if byte_data is not None:
                        audio_int16 = pickle.loads(byte_data)
                        return audio_int16.astype(np.float32) / 32768.0
            
            return np.zeros(self.min_audio_length, dtype=np.float32)
        else:
            from scipy.io import wavfile
            try:
                _, data = wavfile.read(wav_path)
                return data.astype(np.float32) / 32768.0
            except Exception:
                return np.zeros(self.min_audio_length, dtype=np.float32)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        idx1, idx2, label = self.pairs[idx]
        
        audio1 = self._load_audio(self.wav_list[idx1])
        audio2 = self._load_audio(self.wav_list[idx2])
        
        if len(audio1) < self.min_audio_length:
            audio1 = np.pad(audio1, (0, self.min_audio_length - len(audio1)), mode='wrap')
        if len(audio2) < self.min_audio_length:
            audio2 = np.pad(audio2, (0, self.min_audio_length - len(audio2)), mode='wrap')
        
        if len(audio1) > self.max_audio_length:
            start = self.rng.randint(0, len(audio1) - self.max_audio_length)
            audio1 = audio1[start:start + self.max_audio_length]
        if len(audio2) > self.max_audio_length:
            start = self.rng.randint(0, len(audio2) - self.max_audio_length)
            audio2 = audio2[start:start + self.max_audio_length]
        
        return {
            'audio1': torch.from_numpy(audio1).float(),
            'audio2': torch.from_numpy(audio2).float(),
            'label': torch.tensor(label, dtype=torch.float32),
        }
    
    def __getstate__(self):
        state = self.__dict__.copy()
        if self.envs:
            for env in self.envs:
                try:
                    env.close()
                except:
                    pass
        state['envs'] = []
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)


def val_collate_fn(batch):
    """Collate function for validation pairs."""
    max_len1 = max(b['audio1'].shape[0] for b in batch)
    max_len2 = max(b['audio2'].shape[0] for b in batch)
    
    audio1_batch = []
    audio2_batch = []
    labels = []
    
    for b in batch:
        pad_len1 = max_len1 - b['audio1'].shape[0]
        if pad_len1 > 0:
            padded1 = F.pad(b['audio1'], (0, pad_len1))
        else:
            padded1 = b['audio1']
        audio1_batch.append(padded1)
        
        pad_len2 = max_len2 - b['audio2'].shape[0]
        if pad_len2 > 0:
            padded2 = F.pad(b['audio2'], (0, pad_len2))
        else:
            padded2 = b['audio2']
        audio2_batch.append(padded2)
        
        labels.append(b['label'])
    
    return {
        'audio1': torch.stack(audio1_batch),
        'audio2': torch.stack(audio2_batch),
        'label': torch.stack(labels),
    }


# ============================================================
# 3. Model: Finetune Wrapper (same as original)
# ============================================================

class FinetuneableSpeakerEncoder(nn.Module):
    """
    Wrapper for SpeakerEncoder that enables/disables gradient.
    
    Note: Output is 400-dim (same as original), NO projection head.
    This ensures compatibility with P-PhonMatchNet integration.
    """
    
    def __init__(self, model_path: str = 'model/speaker/efficient_tdnn', freeze: bool = False):
        super().__init__()
        
        from model.speaker.encoder import SpeakerEncoder
        
        self.encoder = SpeakerEncoder(
            model_path=model_path,
            freeze=freeze,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        self.output_dim = self.encoder.output_dim
        print(f">> FinetuneableSpeakerEncoder: output_dim = {self.output_dim}")
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [B, T] @ 16kHz
        Returns:
            embedding: [B, output_dim] L2-normalized
        """
        embedding = self.encoder(waveform)
        embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding
    
    def save_checkpoint(self, path: str, epoch: int, eer: float, optimizer_state=None, ge2e_params=None):
        """Save checkpoint in format compatible with P-PhonMatchNet."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.encoder.encoder.state_dict(),
            'eer': eer,
            'embedding_dim': self.output_dim,
        }
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        if ge2e_params is not None:
            checkpoint['ge2e_w'] = ge2e_params['w']
            checkpoint['ge2e_b'] = ge2e_params['b']
        
        torch.save(checkpoint, path)
        print(f">> Saved checkpoint to {path}")


# ============================================================
# 4. GE2E Loss Function
# ============================================================

class GE2ELoss(nn.Module):
    """
    Generalized End-to-End (GE2E) Loss for Speaker Verification
    
    Reference: https://arxiv.org/abs/1710.10467
    
    Key insight: Each utterance should be closest to its own speaker's
    centroid (excluding itself) compared to all other speaker centroids.
    """
    
    def __init__(self, init_w: float = 10.0, init_b: float = -5.0):
        """
        Args:
            init_w: Initial scaling factor
            init_b: Initial bias
        """
        super().__init__()
        
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion = nn.CrossEntropyLoss()
        
        print(f">> GE2ELoss initialized: w={init_w}, b={init_b}")
    
    def forward(
        self,
        embeddings: torch.Tensor,
        n_speakers: int,
        n_utterances: int,
    ) -> tuple:
        """
        Args:
            embeddings: [N*M, D] L2-normalized embeddings
                        N = n_speakers, M = n_utterances
            n_speakers: Number of speakers in batch (N)
            n_utterances: Number of utterances per speaker (M)
        
        Returns:
            loss: GE2E loss
            accuracy: Top-1 accuracy
        """
        device = embeddings.device
        batch_size = n_speakers * n_utterances
        embed_dim = embeddings.shape[-1]
        
        assert embeddings.shape[0] == batch_size, \
            f"Expected {batch_size} embeddings, got {embeddings.shape[0]}"
        
        # Reshape to [N, M, D]
        embeddings = embeddings.view(n_speakers, n_utterances, embed_dim)
        
        # Compute centroids for each speaker [N, D]
        centroids = embeddings.mean(dim=1)
        
        # Build similarity matrix [N, M, N]
        # For each utterance (i, j), compute similarity to all centroids
        # Key: For the "self" centroid, exclude the utterance itself
        cos_sim_matrix = []
        
        for utt_idx in range(n_utterances):
            # Get embeddings for this utterance position across all speakers: [N, D]
            utt_embeddings = embeddings[:, utt_idx, :]
            
            # Compute centroid excluding this utterance for each speaker
            # mask shape: [M] with False at utt_idx
            mask = torch.ones(n_utterances, dtype=torch.bool, device=device)
            mask[utt_idx] = False
            
            # Centroid excluding utterance j: [N, D]
            exc_centroids = embeddings[:, mask, :].mean(dim=1)
            
            # Similarity to own centroid (excluding self): [N]
            cos_sim_self = F.cosine_similarity(utt_embeddings, exc_centroids, dim=-1)
            
            # Similarity to all centroids: [N, N]
            # utt_embeddings: [N, D], centroids: [N, D]
            # We want [N, N] where [i, k] = similarity of speaker i's utterance to speaker k's centroid
            cos_sim_all = F.cosine_similarity(
                utt_embeddings.unsqueeze(1),  # [N, 1, D]
                centroids.unsqueeze(0),         # [1, N, D]
                dim=-1
            )  # [N, N]
            
            # Replace diagonal with self-excluded similarity
            cos_sim_all[range(n_speakers), range(n_speakers)] = cos_sim_self
            
            # Clamp for numerical stability
            cos_sim_all = torch.clamp(cos_sim_all, min=1e-6)
            
            cos_sim_matrix.append(cos_sim_all)
        
        # Stack to [M, N, N] then reshape to [N*M, N]
        cos_sim_matrix = torch.stack(cos_sim_matrix, dim=0)  # [M, N, N]
        cos_sim_matrix = cos_sim_matrix.permute(1, 0, 2).contiguous()  # [N, M, N]
        cos_sim_matrix = cos_sim_matrix.view(-1, n_speakers)  # [N*M, N]
        
        # Apply learnable scaling
        scaled_sim = self.w * cos_sim_matrix + self.b
        
        # Labels: each utterance should match its own speaker
        # Labels are [0, 0, ..., 0 (M times), 1, 1, ..., 1 (M times), ...]
        labels = torch.arange(n_speakers, device=device).repeat_interleave(n_utterances)
        
        # Cross-entropy loss
        loss = self.criterion(scaled_sim, labels)
        
        # Compute accuracy
        predictions = scaled_sim.argmax(dim=-1)
        accuracy = (predictions == labels).float().mean() * 100
        
        return loss, accuracy
    
    def get_params(self):
        """Return current w and b for logging."""
        return {'w': self.w.item(), 'b': self.b.item()}


# ============================================================
# 5. Training and Evaluation
# ============================================================

def train_epoch(
    model, dataloader, optimizer, criterion, device,
    n_speakers, n_utterances, epoch, writer=None, global_step=0
):
    """Train for one epoch with GE2E loss."""
    model.train()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        audio = batch['audio'].to(device)  # [N*M, T]
        
        optimizer.zero_grad()
        
        # Get embeddings
        embeddings = model(audio)  # [N*M, D]
        
        # Compute GE2E loss
        loss, acc = criterion(embeddings, n_speakers, n_utterances)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += acc.item()
        num_batches += 1
        
        # TensorBoard logging
        if writer is not None:
            writer.add_scalar('train/batch_loss', loss.item(), global_step + batch_idx)
            writer.add_scalar('train/batch_acc', acc.item(), global_step + batch_idx)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc.item():.1f}%'})
    
    return total_loss / num_batches, total_acc / num_batches, global_step + len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate and compute EER using pair-based validation."""
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            audio1 = batch['audio1'].to(device)
            audio2 = batch['audio2'].to(device)
            label = batch['label']
            
            emb1 = model(audio1)
            emb2 = model(audio2)
            
            # Cosine similarity
            scores = (emb1 * emb2).sum(dim=-1).cpu()
            
            all_scores.extend(scores.numpy())
            all_labels.extend(label.numpy())
    
    # Compute EER
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    
    # Compute AUC
    roc_auc = auc(fpr, tpr)
    
    return eer * 100, roc_auc * 100


# ============================================================
# 6. Main
# ============================================================

def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # TensorBoard
    writer = None
    if args.use_tensorboard:
        log_dir = os.path.join(args.output_dir, 'logs')
        writer = SummaryWriter(log_dir)
        print(f">> TensorBoard logging to: {log_dir}")
    
    print("=" * 80)
    print("Speaker Encoder Finetuning with GE2E Loss")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Train PKL: {args.train_pkl}")
    print(f"Val PKL: {args.val_pkl}")
    print(f"Epochs: {args.epochs}")
    print(f"N Speakers per batch: {args.n_speakers}")
    print(f"N Utterances per speaker: {args.n_utterances}")
    print(f"Batch size: {args.n_speakers * args.n_utterances}")
    print(f"Learning rate: {args.lr}")
    if args.augment_prob > 0:
        print(f"Noise Augmentation: ENABLED (prob={args.augment_prob}, dir={args.noise_dir})")
    else:
        print(f"Noise Augmentation: DISABLED")
    print("=" * 80)
    
    # Setup audio augmentor (if enabled)
    augmentor = None
    if args.augment_prob > 0 and args.noise_dir:
        augmentor = AudioAugmentor(
            noise_dir=args.noise_dir,
            rir_dir=args.rir_dir,
            snr_range=(args.snr_min, args.snr_max),
            seed=args.seed,
        )
    
    # Training Dataset with GE2E batch sampler
    print("\n>> Creating training dataset...")
    train_dataset = GE2ESpeakerDataset(
        pkl_path=args.train_pkl,
        min_utterances_per_speaker=args.n_utterances,
        augmentor=augmentor,
        augment_prob=args.augment_prob,
    )
    
    train_sampler = GE2EBatchSampler(
        dataset=train_dataset,
        n_speakers=args.n_speakers,
        n_utterances=args.n_utterances,
        num_batches=args.num_train_batches,
        seed=args.seed,
    )
    
    import multiprocessing
    mp_context = multiprocessing.get_context('spawn') if args.num_workers > 0 else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=ge2e_collate_fn,
        multiprocessing_context=mp_context,
        pin_memory=True,
    )
    
    # Validation Dataset (pair-based)
    print("\n>> Creating validation dataset...")
    val_dataset = ValidationPairDataset(
        pkl_path=args.val_pkl,
        num_pairs=args.num_val_pairs,
        seed=args.seed + 1,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=val_collate_fn,
        multiprocessing_context=mp_context,
        pin_memory=True,
    )
    
    # Model
    print("\n>> Loading model...")
    model = FinetuneableSpeakerEncoder(
        model_path=args.speaker_encoder_path,
        freeze=False,
    ).to(device)
    
    # Loss function (GE2E)
    print("\n>> Using GE2E Loss")
    criterion = GE2ELoss(init_w=args.init_w, init_b=args.init_b).to(device)
    
    # Optimizer (include GE2E's w and b parameters)
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': args.lr},
        {'params': criterion.parameters(), 'lr': args.lr * 0.1, 'name': 'ge2e_params'},
    ], weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    # Resume from checkpoint (if specified)
    start_epoch = 1
    if args.resume_from:
        if os.path.exists(args.resume_from):
            print(f"\n>> Resuming from checkpoint: {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
            
            # Load model weights
            model.encoder.encoder.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(">> Restored optimizer state")
            
            # Load GE2E loss parameters if available
            if 'ge2e_w' in checkpoint and 'ge2e_b' in checkpoint:
                criterion.w.data = torch.tensor(checkpoint['ge2e_w'], device=device)
                criterion.b.data = torch.tensor(checkpoint['ge2e_b'], device=device)
                print(f">> Restored GE2E params: w={checkpoint['ge2e_w']:.2f}, b={checkpoint['ge2e_b']:.2f}")
            
            # Load epoch and best EER
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_eer = checkpoint.get('eer', 100.0)
            
            print(f">> Resuming from epoch {start_epoch}")
            print(f">> Previous best EER: {best_eer:.2f}%")
        else:
            print(f"\n>> Warning: Checkpoint not found at {args.resume_from}")
            print(">> Starting from scratch...")
            best_eer = 100.0
    else:
        # Evaluate baseline
        print("\n>> Evaluating baseline (before finetuning)...")
        baseline_eer, baseline_auc = evaluate(model, val_loader, device)
        print(f">> Baseline EER: {baseline_eer:.2f}%, AUC: {baseline_auc:.2f}%")
        best_eer = baseline_eer
        baseline_eer_for_summary = baseline_eer
    
    # Training loop
    print("\n>> Starting training...")
    global_step = 0
    
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc, global_step = train_epoch(
            model, train_loader, optimizer, criterion, device,
            args.n_speakers, args.n_utterances, epoch, writer, global_step
        )
        val_eer, val_auc = evaluate(model, val_loader, device)
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        ge2e_params = criterion.get_params()
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.1f}%")
        print(f"  Val EER: {val_eer:.2f}% | AUC: {val_auc:.2f}%")
        print(f"  LR: {current_lr:.2e}")
        print(f"  GE2E w={ge2e_params['w']:.2f}, b={ge2e_params['b']:.2f}")
        
        # TensorBoard logging
        if writer is not None:
            writer.add_scalar('train/epoch_loss', train_loss, epoch)
            writer.add_scalar('train/epoch_acc', train_acc, epoch)
            writer.add_scalar('val/eer', val_eer, epoch)
            writer.add_scalar('val/auc', val_auc, epoch)
            writer.add_scalar('train/lr', current_lr, epoch)
            writer.add_scalar('ge2e/w', ge2e_params['w'], epoch)
            writer.add_scalar('ge2e/b', ge2e_params['b'], epoch)
        
        # Save best
        if val_eer < best_eer:
            improvement = best_eer - val_eer
            best_eer = val_eer
            
            save_path = os.path.join(args.output_dir, 'best_speaker_encoder.pt')
            model.save_checkpoint(save_path, epoch, val_eer, optimizer.state_dict(), ge2e_params)
            print(f"  ✓ New best! Improved by {improvement:.2f}%")
        
        # Save latest
        latest_path = os.path.join(args.output_dir, 'latest_speaker_encoder.pt')
        model.save_checkpoint(latest_path, epoch, val_eer, optimizer.state_dict(), ge2e_params)
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
    
    # Summary
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    if not args.resume_from:
        print(f"Baseline EER: {baseline_eer_for_summary:.2f}%")
    print(f"Best EER: {best_eer:.2f}%")
    if not args.resume_from:
        print(f"Improvement: {baseline_eer_for_summary - best_eer:.2f}%")
    print(f"\nBest checkpoint: {args.output_dir}/best_speaker_encoder.pt")
    if args.use_tensorboard:
        print(f"TensorBoard logs: {args.output_dir}/logs")
        print(f"  View with: tensorboard --logdir {args.output_dir}/logs")
    print("\nTo use with P-PhonMatchNet:")
    print(f"  python train_personalized.py --finetuned_speaker_encoder {args.output_dir}/best_speaker_encoder.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Speaker Encoder with GE2E Loss")
    
    # Data
    parser.add_argument('--train_pkl', type=str, required=True,
                        help='Path to training pickle file')
    parser.add_argument('--val_pkl', type=str, required=True,
                        help='Path to validation pickle file')
    parser.add_argument('--num_train_batches', type=int, default=1000,
                        help='Number of training batches per epoch')
    parser.add_argument('--num_val_pairs', type=int, default=20000,
                        help='Number of validation pairs')
    
    # Model
    parser.add_argument('--speaker_encoder_path', type=str,
                        default='model/speaker/efficient_tdnn',
                        help='Path to speaker encoder weights')
    
    # GE2E specific
    parser.add_argument('--n_speakers', type=int, default=16,
                        help='Number of speakers per batch (N)')
    parser.add_argument('--n_utterances', type=int, default=10,
                        help='Number of utterances per speaker (M)')
    parser.add_argument('--init_w', type=float, default=10.0,
                        help='Initial w for GE2E loss')
    parser.add_argument('--init_b', type=float, default=-5.0,
                        help='Initial b for GE2E loss')
    
    # Training
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=30)
    
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--output_dir', type=str,
                        default='./checkpoints/speaker_encoder_ge2e',
                        help='Output directory for checkpoints')
    parser.add_argument('--use_tensorboard', action='store_true',
                        help='Enable TensorBoard logging')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume training from (e.g., ./checkpoints/speaker_encoder_ge2e/latest_speaker_encoder.pt)')
    
    # Noise Augmentation
    parser.add_argument('--noise_dir', type=str, default='/padawan/noise/noise_train',
                        help='Path to noise wav files for augmentation')
    parser.add_argument('--rir_dir', type=str, default=None,
                        help='Optional: Path to RIR files for reverberation')
    parser.add_argument('--augment_prob', type=float, default=0.0,
                        help='Probability of applying augmentation per sample (0.0 = off, 0.7 = recommended)')
    parser.add_argument('--snr_min', type=float, default=5.0,
                        help='Minimum SNR (dB) for additive noise')
    parser.add_argument('--snr_max', type=float, default=20.0,
                        help='Maximum SNR (dB) for additive noise')
    
    args = parser.parse_args()
    main(args)
