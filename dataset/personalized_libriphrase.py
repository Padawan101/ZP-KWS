"""
Personalized LibriPhrase Dataset for P-PhonMatchNet Training

Extends the base LibriPhraseDataset with speaker pairing functionality
for personalized keyword spotting.

Features:
- Speaker ID extraction from LibriSpeech-style paths
- Speaker pairing logic (50% same, 50% different by default)
- Enrollment audio selection
- Four test categories: ts-tk, nts-tk, ts-ntk, nts-ntk
"""

import os
import sys
import numpy as np
import torch

# Import base class
from .libriphrase import LibriPhraseDataset


class SubsetDataset(torch.utils.data.Dataset):
    """
    Subset wrapper that shares the parent dataset's data.
    
    This allows creating easy/hard validation sets from a single pkl load,
    reducing memory usage by ~50%.
    
    Args:
        parent_dataset: The base PersonalizedLibriPhraseDataset
        indices: List of indices to include in this subset
    """
    def __init__(self, parent_dataset, indices):
        self.parent = parent_dataset
        self.indices = indices
        # Copy attributes needed for dataloader
        self.batch_size = parent_dataset.batch_size
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Map subset index to parent index
        parent_idx = self.indices[idx]
        return self.parent[parent_idx]
    
    def collate(self, batch):
        return self.parent.collate(batch)
    
    def on_epoch_end(self):
        # Shuffle subset indices
        np.random.shuffle(self.indices)


class PersonalizedLibriPhraseDataset(LibriPhraseDataset):
    """
    Personalized LibriPhrase Dataset for P-PhonMatchNet

    Extends LibriPhraseDataset with speaker verification capabilities.

    Args:
        personalized (bool): Enable personalized mode with enrollment audio
        speaker_ratio (float): Ratio of same-speaker pairs (default: 0.5)
        **kwargs: Arguments passed to LibriPhraseDataset

    Returns (when personalized=True):
        - x: Input audio
        - x_noisy: Noisy input (training only)
        - gemb: Google speech embeddings
        - y: Text features (g2p_embed)
        - z: Keyword label (1=match, 0=mismatch)
        - l: Speech phoneme indices
        - t: Text phoneme indices
        - enrollment_audio: Enrollment audio for speaker verification
        - speaker_label: Speaker match label (1=same, 0=different)
        - final_label: Combined label (keyword AND speaker match)
        - category: One of 'ts-tk', 'nts-tk', 'ts-ntk', 'nts-ntk'
    """

    def __init__(
        self,
        batch_size,
        personalized=False,
        speaker_ratio=0.5,
        max_samples=None,
        seed=42,  # 新增：確保 speaker pairing 可重複
        **kwargs
    ):
        """Initialize personalized dataset"""
        self.seed = seed  # 保存 seed 供 __getitem__ 使用

        # Initialize base class
        super().__init__(batch_size=batch_size, **kwargs)
        
        if max_samples is not None and max_samples > 0:
            original_len = len(self.data)
            self.data = self.data.iloc[:max_samples].reset_index(drop=True)
            self.indices = list(range(len(self.data)))
            self.len = len(self.data)
            print(f">> [Mini Test] Limited dataset: {original_len} → {len(self.data)}")
            if hasattr(self, 'wav_list'):
                self.wav_list = self.wav_list[:max_samples]
            if hasattr(self, 'lab_list'):
                self.lab_list = self.lab_list[:max_samples]
            if hasattr(self, 'idx_list'):
                self.idx_list = self.idx_list[:max_samples]
            if hasattr(self, 'sIdx_list'):
                self.sIdx_list = self.sIdx_list[:max_samples]
            
            # ========== 新增：強制更新 parent 可能用到的變數 ==========
            # 如果 parent 有 shuffle_indices 或其他 index-related 變數
            if hasattr(self, 'shuffle_indices'):
                self.shuffle_indices = list(range(len(self.data)))
    
            # 確保 parent 不會用舊的長度計算
            # 檢查 parent 有沒有存 length 的變數
            if hasattr(self, '_len'):
                self._len = len(self.data)
            if hasattr(self, 'length'):
                self.length = len(self.data)
            # =========================================================
            
            print(f">> [Mini Test] Limited dataset: {original_len} → {len(self.data)}")
            print(f">>   - indices: {len(self.indices)}")
            print(f">>   - wav_list: {len(self.wav_list) if hasattr(self, 'wav_list') else 'N/A'}")
            print(f">>   - lab_list: {len(self.lab_list) if hasattr(self, 'lab_list') else 'N/A'}")

        # Personalized mode settings
        self.personalized = personalized
        self.speaker_ratio = speaker_ratio

        # Prepare speaker information if personalized
        if self.personalized:
            print(">> [Personalized] Preparing speaker information...")
            self._prepare_speaker_info()
            print(f">> [Personalized] Found {len(self.all_speakers)} unique speakers")
            
            # === Memory Optimization: Now safe to delete DataFrame ===
            print(f">> [Memory] Freeing DataFrame (~100MB)...")
            del self.data
            
            # Force garbage collection to ensure memory is freed
            import gc
            gc.collect()
            print(f">> [Memory] Optimization complete. GC done.")
        
        print(f">> [DEBUG] After init:")
        print(f">>   len(self.indices): {len(self.indices)}")
        print(f">>   self.__len__(): {self.__len__()}")
        print(f">>   parent __len__: {super().__len__()}")

    # ========== 關鍵：Override __len__ ==========
    def __len__(self):
        """
        返回實際的 dataset 長度
        確保與 self.indices 同步
        """
        return len(self.indices)
    # ===========================================
    
    def _prepare_speaker_info(self):
        """
        Prepare speaker-related information:
        1. Extract speaker IDs from wav paths
        2. Build speaker → utterances mapping
        3. Get list of all unique speakers
        """

        # 1. Extract speaker IDs and create list FIRST (before any references)
        print(">> [Personalized] Extracting speaker IDs...")
        self.data['speaker_id'] = self.data['wav'].apply(self._extract_speaker_id)
        
        # 2. Extract speaker_id_list immediately (before iterrows which creates references)
        self.speaker_id_list = self.data['speaker_id'].values

        # 3. Build speaker → indices mapping (using extracted list, not DataFrame)
        print(">> [Personalized] Building speaker mappings...")
        self.speaker_to_indices = {}
        for idx, spk in enumerate(self.speaker_id_list):
            if spk not in self.speaker_to_indices:
                self.speaker_to_indices[spk] = []
            self.speaker_to_indices[spk].append(idx)

        # 4. Get all unique speakers
        self.all_speakers = list(self.speaker_to_indices.keys())

        # Statistics
        n_speakers = len(self.all_speakers)
        avg_utts = np.mean([len(v) for v in self.speaker_to_indices.values()])
        print(f">>   - Total speakers: {n_speakers}")
        print(f">>   - Avg utterances per speaker: {avg_utts:.1f}")

    def _extract_speaker_id(self, wav_path):
        """
        Extract speaker ID from LibriSpeech-style path

        Format: {subset}/{speaker_id}/{chapter}/{utterance}.wav
        Or filename: {speaker_id}-{chapter}-{utterance}.wav

        Args:
            wav_path: Path to wav file

        Returns:
            speaker_id: String speaker ID
        """
        # Get filename
        filename = os.path.basename(wav_path)

        # LibriSpeech format: {speaker_id}-{chapter}-{utterance}.wav
        # Remove extension
        filename = filename.replace('.wav', '').replace('.flac', '')

        # Split by '-' and get first part (speaker ID)
        parts = filename.split('-')
        if parts:
            return parts[0]

        # Fallback: use 'unknown'
        return "unknown"

    def _select_enrollment_audio(self, target_speaker, exclude_idx=None, rng=None):
        """
        Select an enrollment utterance for the target speaker

        Args:
            target_speaker: Speaker ID
            exclude_idx: Index to exclude (avoid using same utterance)
            rng: Optional np.random.RandomState for deterministic selection

        Returns:
            enrollment_idx: DataFrame index of enrollment utterance
        """
        if target_speaker not in self.speaker_to_indices:
            # Fallback: use same index if speaker not found
            return exclude_idx if exclude_idx is not None else 0

        # Get all utterances for this speaker
        candidates = self.speaker_to_indices[target_speaker].copy()

        # Exclude current utterance
        if exclude_idx is not None and exclude_idx in candidates:
            candidates.remove(exclude_idx)

        # If no other utterances available, use the same one
        if len(candidates) == 0:
            return exclude_idx if exclude_idx is not None else self.speaker_to_indices[target_speaker][0]

        # Randomly select an enrollment utterance (use rng if provided for determinism)
        if rng is not None:
            return rng.choice(candidates)
        return np.random.choice(candidates)

    def __getitem__(self, idx):
        """
        Get item with optional speaker pairing

        Args:
            idx: Index in dataset

        Returns:
            dict: Data dictionary (format depends on personalized mode)
        """
        # Get base item first
        base_item = super().__getitem__(idx)

        # If not personalized, return base item
        if not self.personalized:
            return base_item

        # === Personalized Mode: Add enrollment audio ===

        i = self.indices[idx]

        # Get speaker info (from pre-extracted list, not DataFrame)
        audio_speaker = self.speaker_id_list[i]
        keyword_label = self.lab_list[i]  # 1=match, 0=mismatch

        # Decide whether to use same speaker
        # === FIX: 使用基於 index 的確定性隨機，確保相同 index 產生相同結果 ===
        item_rng = np.random.RandomState(self.seed + i)
        use_same_speaker = item_rng.random() < self.speaker_ratio

        if use_same_speaker:
            # Same speaker
            enrollment_speaker = audio_speaker
            speaker_label = 1
        else:
            # Different speaker
            other_speakers = [s for s in self.all_speakers if s != audio_speaker]
            if len(other_speakers) > 0:
                enrollment_speaker = item_rng.choice(other_speakers)  # 使用確定性隨機
            else:
                # Fallback: use same speaker if no others available
                enrollment_speaker = audio_speaker
            speaker_label = 0

        # Select enrollment audio (傳入 item_rng 以確保確定性)
        enrollment_idx = self._select_enrollment_audio(enrollment_speaker, exclude_idx=i, rng=item_rng)
        enrollment_audio_np = self._load_wav(self.wav_list[enrollment_idx])
        
        # === FIX: Ensure minimum audio length for EfficientTDNN ===
        # EfficientTDNN's F.pad(..., 'reflect') fails on very short audio
        MIN_SAMPLES = 3200  # 0.2 seconds at 16kHz
        if len(enrollment_audio_np) < MIN_SAMPLES:
            pad_len = MIN_SAMPLES - len(enrollment_audio_np)
            enrollment_audio_np = np.pad(enrollment_audio_np, (0, pad_len), mode='wrap')
        
        enrollment_audio = torch.from_numpy(enrollment_audio_np).float()

        # Calculate final label (only positive if BOTH keyword AND speaker match)
        # This is used for TO-KWS evaluation
        final_label = int(keyword_label == 1 and speaker_label == 1)

        # Determine category
        if keyword_label == 1 and speaker_label == 1:
            category = 'ts-tk'  # target speaker, target keyword
        elif keyword_label == 1 and speaker_label == 0:
            category = 'nts-tk'  # non-target speaker, target keyword
        elif keyword_label == 0 and speaker_label == 1:
            category = 'ts-ntk'  # target speaker, non-target keyword
        else:
            category = 'nts-ntk'  # non-target speaker, non-target keyword

        # Add personalized fields to base item
        base_item['enrollment_audio'] = enrollment_audio
        base_item['speaker_label'] = torch.tensor([speaker_label], dtype=torch.float32)
        base_item['final_label'] = torch.tensor([final_label], dtype=torch.float32)
        base_item['category'] = category

        # === CRITICAL FIX: 分離 keyword label 和 final label ===
        # z_keyword: 純 keyword 匹配標籤（用於 C-KWS loss，不受 speaker 影響）
        # z_final: AND-gated 標籤（用於 TB-KWS / TO-KWS loss）
        # z: 保持為 z_final 以維持向後相容性
        base_item['z_keyword'] = torch.tensor([float(keyword_label)], dtype=torch.float32)
        base_item['z_final'] = torch.tensor([float(final_label)], dtype=torch.float32)
        base_item['z'] = base_item['z_final']  # 預設使用 AND-gated label

        # === NEW: Add type field for easy/hard difficulty tracking ===
        if hasattr(self, 'type_list'):
            base_item['type'] = self.type_list[i]
        else:
            base_item['type'] = 'unknown'

        return base_item

    def collate(self, batch):
        """
        Collate function with enrollment audio support

        Args:
            batch: List of data dicts

        Returns:
            batch_dict: Batched data dictionary
        """
        # Get base collation
        batch_dict = super().collate(batch)

        # If personalized, add enrollment audio fields
        if self.personalized:
            device = batch[0]["x"].device

            # Pad enrollment audio
            enrollment_audios = [b["enrollment_audio"] for b in batch]
            max_enroll_len = max(ea.shape[0] for ea in enrollment_audios)
            batch_dict["enrollment_audio"] = self.pad_sequence(
                enrollment_audios,
                max_enroll_len
            )
            batch_dict["enrollment_len"] = torch.tensor(
                [b["enrollment_audio"].shape[0] for b in batch],
                dtype=torch.int32,
                device=device
            )

            # Speaker and final labels
            batch_dict["speaker_label"] = torch.stack(
                [b["speaker_label"] for b in batch]
            )
            batch_dict["final_label"] = torch.stack(
                [b["final_label"] for b in batch]
            )

            # Categories (list of strings)
            batch_dict["category"] = [b["category"] for b in batch]

            # === NEW: 新增 z_keyword 和 z_final 的 batch 處理 ===
            batch_dict["z_keyword"] = torch.stack([b["z_keyword"] for b in batch])
            batch_dict["z_final"] = torch.stack([b["z_final"] for b in batch])

            # === NEW: Add type field for easy/hard difficulty tracking ===
            if 'type' in batch[0]:
                batch_dict["type"] = [b["type"] for b in batch]

        return batch_dict


def create_personalized_dataloaders(args, train_personalized=False):
    """
    Create dataloaders for P-PhonMatchNet training

    Args:
        args: Training arguments
        train_personalized: Whether to use personalized training

    Returns:
        train_loader: Training dataloader
        val_loaders: List of validation dataloaders
        vocab: Vocabulary size
        train_len: Training dataset length
    """
    from dataset import KWSDataLoader

    # Google embedding directory
    if args.audio_input == "raw":
        gemb_dir = None
    else:
        gemb_dir = '/padawan/google_speech_embedding/DB/'

    # Training dataset (personalized if requested)
    train_dataset = PersonalizedLibriPhraseDataset(
        batch_size=args.batch_size,
        gemb_dir=gemb_dir,
        features=args.text_input,
        train=True,
        types='both',
        shuffle=True,
        pkl=args.train_pkl,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
        audio_noise=getattr(args, 'audio_noise', False),
        personalized=train_personalized,
        speaker_ratio=getattr(args, 'speaker_ratio', 0.5),
        max_samples=getattr(args, 'max_train_samples', None),
        frame_labels_path=getattr(args, 'frame_labels_path', None),  # MFA Aux CE
    )
    
    is_multiprocessing = (args.num_workers > 0)
    train_loader = KWSDataLoader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=args.num_workers,
        prefetch_factor=(2 if is_multiprocessing else None),
        persistent_workers=is_multiprocessing
    )
    
    # ========== Debug DataLoader ==========
    print(f"\n>> [DEBUG] DataLoader Info:")
    print(f">>   Training:")
    print(f">>     - len(train_dataset): {len(train_dataset)}")
    print(f">>     - batch_size: {args.batch_size}")
    print(f">>     - len(train_loader): {len(train_loader)}")
    print(f">>     - Expected steps: {len(train_dataset) // args.batch_size + (1 if len(train_dataset) % args.batch_size else 0)}")
    if hasattr(train_loader, 'sampler'):
        print(f">>     - sampler type: {type(train_loader.sampler)}")
        if hasattr(train_loader.sampler, '__len__'):
            print(f">>     - len(sampler): {len(train_loader.sampler)}")
    # ======================================

    # ========== Validation datasets (optimized: single pkl load) ==========
    # Load pkl once with types='both', then create subset views
    print("\n>> [Optimization] Loading validation pkl once, creating easy/hard subsets...")
    
    base_val_dataset = PersonalizedLibriPhraseDataset(
        batch_size=args.batch_size,
        gemb_dir=gemb_dir,
        features=args.text_input,
        train=False,
        types='both',  # Load both types
        shuffle=False,  # CRITICAL FIX: 必須 False，確保 DataFrame index 與 Dataset index 對應
        pkl=args.libriphrase_pkl,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
        personalized=train_personalized,
        speaker_ratio=getattr(args, 'speaker_ratio', 0.5),
        max_samples=getattr(args, 'max_val_samples', None)
    )
    
    # Build indices for easy and hard subsets
    easy_indices = []
    hard_indices = []
    for i, sample_type in enumerate(base_val_dataset.type_list):
        if 'easy' in str(sample_type):
            easy_indices.append(i)
        elif 'hard' in str(sample_type):
            hard_indices.append(i)
    
    print(f">>   - Total samples: {len(base_val_dataset)}")
    print(f">>   - Easy subset: {len(easy_indices)} samples")
    print(f">>   - Hard subset: {len(hard_indices)} samples")
    
    # Create subset views (share underlying data)
    val_easy_dataset = SubsetDataset(base_val_dataset, easy_indices)
    val_hard_dataset = SubsetDataset(base_val_dataset, hard_indices)

    # Create dataloaders
    val_easy_loader = KWSDataLoader(
        val_easy_dataset,
        args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=args.num_workers
    )

    val_hard_loader = KWSDataLoader(
        val_hard_dataset,
        args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=args.num_workers
    )

    val_loaders = [val_easy_loader, val_hard_loader]

    vocab = train_dataset.nPhoneme
    train_len = len(train_dataset)

    return train_loader, val_loaders, vocab, train_len
