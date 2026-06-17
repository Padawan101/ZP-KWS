#!/usr/bin/env python3
"""
find_oov_words.py - 手動檢測 LAB 檔案中的 OOV 詞彙

Usage:
    python find_oov_words.py --corpus_dir /padawan/mfa_staging
"""

import argparse
from pathlib import Path
from collections import Counter


def load_mfa_dict(dict_path: str = None) -> set:
    """
    載入 MFA 字典（如果可用）
    如果沒有字典檔案，返回空集合（無法判斷 OOV）
    """
    if dict_path is None or not Path(dict_path).exists():
        print(f"Warning: Dictionary not provided or not found at {dict_path}")
        return set()
    
    vocab = set()
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # MFA format: WORD phoneme1 phoneme2 ...
            word = line.split()[0].lower()
            vocab.add(word)
    
    return vocab


def collect_words_from_labs(corpus_dir: str) -> Counter:
    """
    從所有 .lab 檔案中收集詞彙並計數
    """
    corpus_path = Path(corpus_dir)
    word_counter = Counter()
    
    for lab_file in corpus_path.rglob("*.lab"):
        with open(lab_file, 'r', encoding='utf-8') as f:
            text = f.read().strip().lower()
            words = text.split()
            word_counter.update(words)
    
    return word_counter


def main():
    parser = argparse.ArgumentParser(description='Find OOV words in corpus')
    parser.add_argument('--corpus_dir', type=str, required=True,
                        help='Path to MFA staging directory')
    parser.add_argument('--dict_path', type=str, default=None,
                        help='Path to MFA dictionary file (optional)')
    parser.add_argument('--output', type=str, default='oov_analysis.txt',
                        help='Output file for OOV report')
    args = parser.parse_args()
    
    print("=" * 60)
    print("OOV Word Detection")
    print("=" * 60)
    
    # Step 1: Collect all words
    print(f"\n[Step 1] Collecting words from {args.corpus_dir}...")
    word_counter = collect_words_from_labs(args.corpus_dir)
    print(f"Found {sum(word_counter.values())} total tokens")
    print(f"Found {len(word_counter)} unique word types")
    
    # Step 2: Load dictionary (if provided)
    if args.dict_path:
        print(f"\n[Step 2] Loading dictionary from {args.dict_path}...")
        vocab = load_mfa_dict(args.dict_path)
        print(f"Dictionary contains {len(vocab)} words")
        
        # Find OOVs
        oov_words = {word: count for word, count in word_counter.items() 
                     if word not in vocab}
        
        print(f"\n[Step 3] OOV Analysis:")
        print(f"  OOV word types: {len(oov_words)}")
        print(f"  OOV tokens: {sum(oov_words.values())}")
        print(f"  OOV rate: {sum(oov_words.values()) / sum(word_counter.values()) * 100:.2f}%")
        
        # Save report
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write("OOV Words (sorted by frequency)\n")
            f.write("=" * 60 + "\n\n")
            
            for word, count in sorted(oov_words.items(), key=lambda x: -x[1]):
                f.write(f"{word:30s} {count:8d}\n")
        
        print(f"\nOOV report saved to: {args.output}")
        
        # Print top 20 OOVs
        print("\nTop 20 OOV words:")
        for word, count in sorted(oov_words.items(), key=lambda x: -x[1])[:20]:
            print(f"  {word:30s} {count:8d}")
    
    else:
        # No dictionary - just list all words
        print("\n[Step 2] No dictionary provided - listing all words")
        
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write("All Words (sorted by frequency)\n")
            f.write("=" * 60 + "\n\n")
            
            for word, count in sorted(word_counter.items(), key=lambda x: -x[1]):
                f.write(f"{word:30s} {count:8d}\n")
        
        print(f"\nWord list saved to: {args.output}")
        
        # Print top 50
        print("\nTop 50 words:")
        for word, count in sorted(word_counter.items(), key=lambda x: -x[1])[:50]:
            print(f"  {word:30s} {count:8d}")


if __name__ == '__main__':
    main()
