#!/usr/bin/env python3
"""
prepare_mfa_from_lmdb.py - Extract audio files from LMDB for MFA alignment

        
Output structure:
    mfa_staging/
    ├── 1034/
    │   ├── 1034-121119-0075_1word_0.wav
    │   ├── 1034-121119-0075_1word_0.lab  # content: "madame"
    │   └── ...
    └── ...
"""

import argparse
import os
import pickle
from pathlib import Path
from collections import defaultdict

import lmdb
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Extract audio from LMDB for MFA')
    parser.add_argument('--csv_path', type=str, nargs='+', required=True,
                        help='Path(s) to LibriPhrase CSV file(s)')
    parser.add_argument('--lmdb_path', type=str, nargs='+', required=True,
                        help='Path(s) to LMDB database(s)')
    parser.add_argument('--output_dir', type=str, default='./mfa_staging',
                        help='Output directory for MFA files')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Target sample rate (default: 16000)')
    return parser.parse_args()


def collect_unique_files(csv_paths: list) -> dict:
    """
    Collect unique (path, speaker, text) tuples from CSV files.
    Returns: dict[path] = (speaker_id, text)
    """
    unique_files = {}
    
    for csv_path in csv_paths:
        print(f"Reading {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Collect from anchor columns
        for _, row in df.iterrows():
            anchor_path = row['anchor']
            anchor_spk = str(row['anchor_spk'])
            anchor_text = row['anchor_text'].lower().strip()
            
            if anchor_path not in unique_files:
                unique_files[anchor_path] = (anchor_spk, anchor_text)
            
            # Collect from comparison columns
            comp_path = row['comparison']
            comp_spk = str(row['comparison_spk'])
            comp_text = row['comparison_text'].lower().strip()
            
            if comp_path not in unique_files:
                unique_files[comp_path] = (comp_spk, comp_text)
    
    return unique_files


def extract_files(unique_files: dict, lmdb_paths: list, output_dir: str, sample_rate: int):
    """
    Extract audio files from LMDB and create .wav + .lab files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Open all LMDB environments
    envs = []
    for lmdb_path in lmdb_paths:
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        envs.append(env)
    
    success_count = 0
    failed_files = []
    speakers = set()
    
    for file_path, (speaker_id, text) in tqdm(unique_files.items(), desc="Extracting"):
        # Create speaker directory
        speaker_dir = output_path / speaker_id
        speaker_dir.mkdir(exist_ok=True)
        speakers.add(speaker_id)
        
        # Get filename without directory
        filename = Path(file_path).stem  # e.g., "1034-121119-0075_1word_0"
        
        # Try to find the key in any LMDB
        audio_data = None
        for env in envs:
            with env.begin() as txn:
                value = txn.get(file_path.encode('utf-8'))
                if value is not None:
                    try:
                        audio_data = pickle.loads(value)
                        break
                    except Exception as e:
                        print(f"Warning: Failed to unpickle {file_path}: {e}")
                        continue
        
        if audio_data is None:
            failed_files.append(file_path)
            continue
        
        # Audio is int16 numpy array
        if audio_data.dtype == np.int16:
            # Convert to float32 for saving
            audio_float = audio_data.astype(np.float32) / 32768.0
        else:
            audio_float = audio_data.astype(np.float32)
        
        # Save WAV file
        wav_path = speaker_dir / f"{filename}.wav"
        sf.write(wav_path, audio_float, sample_rate, subtype='PCM_16')
        
        # Save lab file (lowercase text)
        lab_path = speaker_dir / f"{filename}.lab"
        with open(lab_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        success_count += 1
    
    # Close all LMDB environments
    for env in envs:
        env.close()
    
    return success_count, failed_files, speakers


def main():
    args = parse_args()
    
    print("=" * 60)
    print("MFA Dataset Preparation from LMDB")
    print("=" * 60)
    
    # Step 1: Collect unique files
    print("\n[Step 1] Collecting unique files from CSV...")
    unique_files = collect_unique_files(args.csv_path)
    print(f"Found {len(unique_files)} unique audio files")
    
    # Step 2: Extract files
    print(f"\n[Step 2] Extracting to {args.output_dir}...")
    success, failed, speakers = extract_files(
        unique_files, 
        args.lmdb_path, 
        args.output_dir, 
        args.sample_rate
    )
    
    # Step 3: Report
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total unique files: {len(unique_files)}")
    print(f"Successfully extracted: {success}")
    print(f"Failed: {len(failed)}")
    print(f"Unique speakers: {len(speakers)}")
    print(f"Output directory: {args.output_dir}")
    
    if failed:
        print(f"\nFailed files (first 10):")
        for f in failed[:10]:
            print(f"  - {f}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
        
        # Save failed list
        failed_path = Path(args.output_dir) / "failed_files.txt"
        with open(failed_path, 'w') as f:
            for path in failed:
                f.write(f"{path}\n")
        print(f"\nFull failed list saved to: {failed_path}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
