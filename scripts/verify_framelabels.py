#!/usr/bin/env python3
"""
verify_framelabels.py

驗證 frame_labels.lmdb 的正確性，包含覆蓋率、長度一致性、音素分佈，並產生視覺化圖片。
根據 MFA_AuxCE_Design.md Step 3 設計。

"""

import argparse
import json
import lmdb
import math
import os
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Try to import visualization libraries (optional)
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualizations")

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("Warning: librosa not available, skipping spectrograms")


# Frame shift
FRAME_SHIFT_SEC = 0.020

# Reverse phoneme vocab for visualization
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


def load_lmdb_labels(lmdb_path: str) -> Dict[str, np.ndarray]:
    """
    從 LMDB 載入所有 frame labels。
    
    Returns:
        dict: {file_id: np.array of phone_ids}
    """
    labels = {}
    env = lmdb.open(lmdb_path, readonly=True)
    
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            key_str = key.decode('utf-8')
            # Skip metadata keys
            if key_str.startswith('__'):
                continue
            labels[key_str] = pickle.loads(value)
    
    env.close()
    return labels


def load_lmdb_metadata(lmdb_path: str) -> dict:
    """
    從 LMDB 載入 metadata。
    """
    metadata = {}
    env = lmdb.open(lmdb_path, readonly=True)
    
    with env.begin() as txn:
        for key_name in ['__phoneme_vocab__', '__n_phonemes__', '__frame_shift_sec__']:
            value = txn.get(key_name.encode('utf-8'))
            if value:
                metadata[key_name] = pickle.loads(value)
    
    env.close()
    return metadata


def verify_coverage(labels: Dict[str, np.ndarray], textgrid_dir: str) -> dict:
    """
    驗證覆蓋率：所有 TextGrid 是否都有對應的 frame labels。
    """
    textgrid_dir = Path(textgrid_dir)
    textgrid_files = list(textgrid_dir.glob("**/*.TextGrid"))
    
    textgrid_ids = set()
    for tg_path in textgrid_files:
        file_id = tg_path.stem  # Remove .TextGrid extension
        textgrid_ids.add(file_id)
    
    label_ids = set(labels.keys())
    
    # Find missing
    missing_in_lmdb = textgrid_ids - label_ids
    extra_in_lmdb = label_ids - textgrid_ids
    
    return {
        "total_textgrids": len(textgrid_ids),
        "total_labels": len(label_ids),
        "coverage_rate": len(label_ids) / len(textgrid_ids) * 100 if textgrid_ids else 0,
        "missing_in_lmdb": len(missing_in_lmdb),
        "extra_in_lmdb": len(extra_in_lmdb),
        "missing_samples": list(missing_in_lmdb)[:10],
    }


def verify_length_consistency(labels: Dict[str, np.ndarray], wav_dir: str, 
                               sample_size: int = 100) -> dict:
    """
    驗證 frame label 長度與 audio 預期長度的一致性。
    
    預期長度 = ceil(audio_duration_sec / 0.020)
    
    允許 ±2 frame tolerance（因為 Conv padding 等因素）
    """
    if not HAS_LIBROSA:
        return {"error": "librosa not available"}
    
    wav_dir = Path(wav_dir)
    
    # Sample some files
    file_ids = list(labels.keys())
    if len(file_ids) > sample_size:
        file_ids = random.sample(file_ids, sample_size)
    
    length_diffs = []
    mismatches = []
    
    for file_id in file_ids:
        # Try to find the corresponding wav file
        # file_id format: {speaker}-{chapter}-{utt}_{nword}_word_{idx}
        parts = file_id.split('-')
        if len(parts) >= 2:
            speaker_id = parts[0]
            wav_pattern = f"{speaker_id}/**/{file_id}.wav"
            wav_files = list(wav_dir.glob(wav_pattern))
            
            if wav_files:
                wav_path = wav_files[0]
                try:
                    # Get audio duration
                    duration = librosa.get_duration(path=str(wav_path))
                    expected_frames = math.ceil(duration / FRAME_SHIFT_SEC)
                    actual_frames = len(labels[file_id])
                    
                    diff = actual_frames - expected_frames
                    length_diffs.append(diff)
                    
                    if abs(diff) > 2:
                        mismatches.append({
                            "file_id": file_id,
                            "expected": expected_frames,
                            "actual": actual_frames,
                            "diff": diff
                        })
                except Exception as e:
                    pass
    
    return {
        "samples_checked": len(length_diffs),
        "mean_diff": np.mean(length_diffs) if length_diffs else 0,
        "std_diff": np.std(length_diffs) if length_diffs else 0,
        "max_diff": max(length_diffs) if length_diffs else 0,
        "min_diff": min(length_diffs) if length_diffs else 0,
        "mismatches_gt2": len(mismatches),
        "mismatch_samples": mismatches[:5],
    }


def verify_phoneme_distribution(labels: Dict[str, np.ndarray], 
                                 phoneme_vocab: Dict[str, int]) -> dict:
    """
    統計音素分佈。
    """
    # Reverse vocab
    idx_to_phoneme = {v: k for k, v in phoneme_vocab.items()}
    
    # Count all phonemes
    phone_counter = Counter()
    total_frames = 0
    
    for file_id, frame_labels in labels.items():
        for phone_id in frame_labels:
            phone_counter[phone_id] += 1
        total_frames += len(frame_labels)
    
    # Convert to phoneme names and percentages
    distribution = {}
    for phone_id, count in phone_counter.most_common():
        phoneme = idx_to_phoneme.get(phone_id, f"UNK_{phone_id}")
        distribution[phoneme] = {
            "count": count,
            "percentage": count / total_frames * 100
        }
    
    # Check SIL percentage
    sil_pct = distribution.get("SIL", {}).get("percentage", 0)
    pad_pct = distribution.get("<PAD>", {}).get("percentage", 0)
    
    return {
        "total_frames": total_frames,
        "unique_phonemes": len(phone_counter),
        "sil_percentage": sil_pct,
        "pad_percentage": pad_pct,
        "warning_high_silence": sil_pct > 50,
        "distribution": distribution,
    }


def visualize_samples(labels: Dict[str, np.ndarray], 
                      wav_dir: str,
                      phoneme_vocab: Dict[str, int],
                      output_dir: str,
                      n_samples: int = 10):
    """
    抽樣視覺化：繪製 spectrogram + frame label overlay。
    """
    if not HAS_MATPLOTLIB or not HAS_LIBROSA:
        print("Skipping visualization (missing matplotlib or librosa)")
        return
    
    # Reverse vocab
    idx_to_phoneme = {v: k for k, v in phoneme_vocab.items()}
    
    wav_dir = Path(wav_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample files
    file_ids = list(labels.keys())
    sample_ids = random.sample(file_ids, min(n_samples, len(file_ids)))
    
    for file_id in sample_ids:
        # Find wav file
        parts = file_id.split('-')
        if len(parts) >= 2:
            speaker_id = parts[0]
            wav_files = list(wav_dir.glob(f"{speaker_id}/**/{file_id}.wav"))
            
            if not wav_files:
                continue
            
            wav_path = wav_files[0]
            
            try:
                # Load audio
                y, sr = librosa.load(str(wav_path), sr=16000)
                
                # Compute mel spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=y, sr=sr, n_fft=400, hop_length=160, n_mels=80
                )
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                # Get frame labels
                frame_labels = labels[file_id]
                
                # Create visualization
                fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
                
                # Spectrogram
                img = librosa.display.specshow(
                    mel_spec_db, sr=sr, hop_length=160, 
                    x_axis='time', y_axis='mel', ax=axes[0]
                )
                axes[0].set_title(f"Mel Spectrogram: {file_id}")
                fig.colorbar(img, ax=axes[0], format='%+2.0f dB')
                
                # Frame labels
                # Convert frame indices to time
                times = np.arange(len(frame_labels)) * FRAME_SHIFT_SEC
                
                # Get phoneme names
                phoneme_names = [idx_to_phoneme.get(pid, '?') for pid in frame_labels]
                
                # Plot as step function
                axes[1].step(times, frame_labels, where='mid', linewidth=1)
                axes[1].set_ylabel('Phoneme ID')
                axes[1].set_xlabel('Time (s)')
                axes[1].set_title('Frame-level Phoneme Labels')
                axes[1].set_ylim(-1, max(phoneme_vocab.values()) + 1)
                
                # Add phoneme annotations for changes
                prev_pid = -1
                for i, pid in enumerate(frame_labels):
                    if pid != prev_pid:
                        phoneme = idx_to_phoneme.get(pid, '?')
                        if phoneme not in ['<PAD>', 'SIL']:
                            axes[1].annotate(
                                phoneme, 
                                (times[i], pid),
                                fontsize=6,
                                rotation=45
                            )
                        prev_pid = pid
                
                plt.tight_layout()
                plt.savefig(output_dir / f"{file_id}.png", dpi=150)
                plt.close()
                
                print(f"  Saved: {file_id}.png")
                
            except Exception as e:
                print(f"  Error visualizing {file_id}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify frame_labels.lmdb correctness"
    )
    parser.add_argument(
        "--lmdb_path",
        type=str,
        required=True,
        help="Path to frame_labels.lmdb"
    )
    parser.add_argument(
        "--textgrid_dir",
        type=str,
        default=None,
        help="TextGrid directory for coverage check"
    )
    parser.add_argument(
        "--wav_dir",
        type=str,
        default=None,
        help="WAV directory for length check and visualization"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for visualizations and report"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of samples to visualize"
    )
    args = parser.parse_args()
    
    print(f"Loading LMDB: {args.lmdb_path}")
    labels = load_lmdb_labels(args.lmdb_path)
    metadata = load_lmdb_metadata(args.lmdb_path)
    
    phoneme_vocab = metadata.get('__phoneme_vocab__', DEFAULT_PHONEME_VOCAB)
    n_phonemes = metadata.get('__n_phonemes__', len(DEFAULT_PHONEME_VOCAB))
    
    print(f"\n{'='*60}")
    print("LMDB Statistics")
    print(f"{'='*60}")
    print(f"  Total entries: {len(labels):,}")
    print(f"  Phoneme vocab size: {n_phonemes}")
    
    report = {
        "lmdb_path": args.lmdb_path,
        "total_entries": len(labels),
        "n_phonemes": n_phonemes,
    }
    
    # 1. Coverage check
    if args.textgrid_dir:
        print(f"\n{'='*60}")
        print("Coverage Check")
        print(f"{'='*60}")
        coverage = verify_coverage(labels, args.textgrid_dir)
        report["coverage"] = coverage
        print(f"  TextGrid files: {coverage['total_textgrids']:,}")
        print(f"  Labels in LMDB: {coverage['total_labels']:,}")
        print(f"  Coverage rate: {coverage['coverage_rate']:.2f}%")
        if coverage['missing_in_lmdb'] > 0:
            print(f"  ⚠️  Missing in LMDB: {coverage['missing_in_lmdb']}")
    
    # 2. Phoneme distribution
    print(f"\n{'='*60}")
    print("Phoneme Distribution")
    print(f"{'='*60}")
    dist = verify_phoneme_distribution(labels, phoneme_vocab)
    report["distribution"] = dist
    print(f"  Total frames: {dist['total_frames']:,}")
    print(f"  Unique phonemes: {dist['unique_phonemes']}")
    print(f"  SIL percentage: {dist['sil_percentage']:.2f}%")
    if dist['warning_high_silence']:
        print(f"  ⚠️  Warning: SIL > 50%!")
    
    print("\n  Top 15 phonemes:")
    for i, (phoneme, info) in enumerate(dist['distribution'].items()):
        if i >= 15:
            break
        print(f"    {phoneme:6s}: {info['count']:>10,} ({info['percentage']:5.2f}%)")
    
    # 3. Length consistency (if wav_dir provided)
    if args.wav_dir:
        print(f"\n{'='*60}")
        print("Length Consistency Check")
        print(f"{'='*60}")
        length_check = verify_length_consistency(labels, args.wav_dir)
        report["length_check"] = length_check
        if "error" not in length_check:
            print(f"  Samples checked: {length_check['samples_checked']}")
            print(f"  Mean diff: {length_check['mean_diff']:.2f} frames")
            print(f"  Std diff: {length_check['std_diff']:.2f} frames")
            print(f"  Mismatches (>2 frames): {length_check['mismatches_gt2']}")
    
    # 4. Visualization
    if args.output_dir and args.wav_dir:
        print(f"\n{'='*60}")
        print("Generating Visualizations")
        print(f"{'='*60}")
        visualize_samples(
            labels=labels,
            wav_dir=args.wav_dir,
            phoneme_vocab=phoneme_vocab,
            output_dir=args.output_dir,
            n_samples=args.n_samples
        )
        
        # Save report
        report_path = Path(args.output_dir) / "verification_report.json"
        with open(report_path, 'w') as f:
            # Remove non-serializable items from distribution
            report_serializable = report.copy()
            if 'distribution' in report_serializable:
                report_serializable['distribution'] = {
                    k: v for k, v in report_serializable['distribution'].items()
                    if k != 'distribution'
                }
            json.dump(report_serializable, f, indent=2, default=str)
        print(f"\n✓ Report saved to: {report_path}")
    
    print(f"\n{'='*60}")
    print("Verification Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
