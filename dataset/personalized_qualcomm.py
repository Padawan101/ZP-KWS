"""
Personalized Qualcomm Keyword Speech Dataset

Extends QualcommKeywordSpeechDataset with speaker pairing for P-KWS evaluation.

Speaker ID is extracted from directory structure:
    {base_dir}/{keyword}/{speaker_dir}/{utterance}.wav
    Example: /home/DB/qualcomm/hey_snapdragon/M001/001.wav → "M001"
"""

import os
import numpy as np
import torch
from .qualcomm import QualcommKeywordSpeechDataset


class PersonalizedQualcommDataset(QualcommKeywordSpeechDataset):
    """
    Qualcomm dataset with speaker pairing support for P-KWS evaluation.
    
    Args:
        batch_size: Batch size
        personalized: Enable speaker pairing
        speaker_ratio: Ratio of same-speaker pairs (default: 0.5)
        **kwargs: Arguments passed to QualcommKeywordSpeechDataset
    """
    
    def __init__(
        self,
        batch_size,
        personalized=False,
        speaker_ratio=0.5,
        **kwargs
    ):
        super().__init__(batch_size=batch_size, **kwargs)
        
        self.personalized = personalized
        self.speaker_ratio = speaker_ratio
        
        if self.personalized:
            self._prepare_speaker_info()
    
    def _extract_speaker_id(self, wav_path):
        """
        Extract speaker ID from Qualcomm path.
        
        Format: {base_dir}/{keyword}/{speaker_dir}/{utterance}.wav
        Example: /home/DB/qualcomm/hey_snapdragon/M001/001.wav → "M001"
        """
        parts = wav_path.split(os.sep)
        if len(parts) >= 2:
            return parts[-2]  # speaker directory
        return "unknown"
    
    def _prepare_speaker_info(self):
        """Build speaker → indices mapping"""
        print(">> [Personalized Qualcomm] Extracting speaker IDs...")
        
        self.speaker_ids = [self._extract_speaker_id(w) for w in self.wav_list]
        
        self.speaker_to_indices = {}
        for idx, spk in enumerate(self.speaker_ids):
            if spk not in self.speaker_to_indices:
                self.speaker_to_indices[spk] = []
            self.speaker_to_indices[spk].append(idx)
        
        self.all_speakers = list(self.speaker_to_indices.keys())
        
        n_speakers = len(self.all_speakers)
        avg_utts = np.mean([len(v) for v in self.speaker_to_indices.values()])
        print(f">>   Total speakers: {n_speakers}")
        print(f">>   Avg utterances per speaker: {avg_utts:.1f}")
    
    def _select_enrollment_audio(self, target_speaker, exclude_idx=None):
        """Select an enrollment utterance for target speaker"""
        if target_speaker not in self.speaker_to_indices:
            return exclude_idx if exclude_idx is not None else 0
        
        candidates = self.speaker_to_indices[target_speaker].copy()
        
        if exclude_idx is not None and exclude_idx in candidates:
            candidates.remove(exclude_idx)
        
        if len(candidates) == 0:
            return exclude_idx if exclude_idx is not None else \
                   self.speaker_to_indices[target_speaker][0]
        
        return np.random.choice(candidates)
    
    def __getitem__(self, idx):
        """Get item with optional speaker pairing"""
        base_item = super().__getitem__(idx)
        
        if not self.personalized:
            return base_item
        
        i = self.indices[idx]
        audio_speaker = self.speaker_ids[i]
        keyword_label = self.lab_list[i]
        
        use_same_speaker = np.random.random() < self.speaker_ratio
        
        if use_same_speaker:
            enrollment_speaker = audio_speaker
            speaker_label = 1
        else:
            other_speakers = [s for s in self.all_speakers if s != audio_speaker]
            if other_speakers:
                enrollment_speaker = np.random.choice(other_speakers)
            else:
                enrollment_speaker = audio_speaker
            speaker_label = 0
        
        enrollment_idx = self._select_enrollment_audio(
            enrollment_speaker, exclude_idx=i
        )
        enrollment_audio_np = self._load_wav(self.wav_list[enrollment_idx])
        
        # === FIX: Ensure minimum audio length for EfficientTDNN ===
        # EfficientTDNN's F.pad(..., 'reflect') fails on very short audio
        MIN_SAMPLES = 3200  # 0.2 seconds at 16kHz
        if len(enrollment_audio_np) < MIN_SAMPLES:
            pad_len = MIN_SAMPLES - len(enrollment_audio_np)
            enrollment_audio_np = np.pad(enrollment_audio_np, (0, pad_len), mode='wrap')
        
        enrollment_audio = torch.from_numpy(enrollment_audio_np).float()
        
        final_label = int(keyword_label == 1 and speaker_label == 1)
        
        if keyword_label == 1 and speaker_label == 1:
            category = 'ts-tk'
        elif keyword_label == 1 and speaker_label == 0:
            category = 'nts-tk'
        elif keyword_label == 0 and speaker_label == 1:
            category = 'ts-ntk'
        else:
            category = 'nts-ntk'
        
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

        return base_item
    
    def collate(self, batch):
        """Collate with enrollment audio support"""
        batch_dict = super().collate(batch)
        
        if self.personalized:
            device = batch[0]["x"].device
            
            enrollment_audios = [b["enrollment_audio"] for b in batch]
            max_len = max(ea.shape[0] for ea in enrollment_audios)
            batch_dict["enrollment_audio"] = self.pad_sequence(
                enrollment_audios, max_len
            )
            batch_dict["enrollment_len"] = torch.tensor(
                [b["enrollment_audio"].shape[0] for b in batch],
                dtype=torch.int32, device=device
            )
            
            batch_dict["speaker_label"] = torch.stack(
                [b["speaker_label"] for b in batch]
            )
            batch_dict["final_label"] = torch.stack(
                [b["final_label"] for b in batch]
            )
            batch_dict["category"] = [b["category"] for b in batch]

            # === NEW: 新增 z_keyword 和 z_final 的 batch 處理 ===
            batch_dict["z_keyword"] = torch.stack([b["z_keyword"] for b in batch])
            batch_dict["z_final"] = torch.stack([b["z_final"] for b in batch])
        
        return batch_dict