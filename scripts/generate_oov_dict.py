#!/usr/bin/env python3
"""
generate_oov_dict.py - Use existing g2p model to generate pronunciations for OOV words

"""

import argparse
import sys
from pathlib import Path

# Add g2p to path
sys.path.insert(0, '/padawan/PKWS/dataset/g2p')
from g2p_en import G2p


def parse_args():
    parser = argparse.ArgumentParser(description='Generate pronunciations for OOV words')
    parser.add_argument('--oov_file', type=str, required=True,
                        help='Path to OOV analysis file')
    parser.add_argument('--g2p_path', type=str, 
                        default='/padawan/PKWS/dataset/g2p/g2p_en',
                        help='Path to g2p model')
    parser.add_argument('--output', type=str, required=True,
                        help='Output dictionary file')
    parser.add_argument('--top_n', type=int, default=500,
                        help='Only generate for top N OOV words')
    return parser.parse_args()


def load_oov_words(oov_file: str, top_n: int = 500) -> list:
    """
    Load OOV words from analysis file
    Returns: [(word, count), ...]
    """
    words = []
    with open(oov_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('=') or line.startswith('OOV'):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                word = parts[0]
                try:
                    count = int(parts[1])
                    words.append((word, count))
                except ValueError:
                    continue
    
    # Sort by count and take top N
    words.sort(key=lambda x: -x[1])
    return words[:top_n]


def main():
    args = parse_args()
    
    print("=" * 60)
    print("OOV Dictionary Generation")
    print("=" * 60)
    
    # Load G2P model
    print(f"\n[Step 1] Loading G2P model from {args.g2p_path}...")
    g2p = G2p()
    
    # Load OOV words
    print(f"\n[Step 2] Loading OOV words from {args.oov_file}...")
    oov_words = load_oov_words(args.oov_file, args.top_n)
    print(f"Loaded {len(oov_words)} OOV words")
    
    # Generate pronunciations
    print(f"\n[Step 3] Generating pronunciations...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for word, count in oov_words:
            # Get phoneme sequence
            phonemes = g2p(word)
            
            # Filter out non-phoneme symbols (like punctuation)
            phonemes = [p for p in phonemes if p.isalpha() or p in ['0', '1', '2']]
            
            if phonemes:
                # MFA format: WORD phoneme1 phoneme2 ...
                phoneme_str = ' '.join(phonemes).upper()
                f.write(f"{word} {phoneme_str}\n")
                
                if (oov_words.index((word, count)) + 1) % 100 == 0:
                    print(f"  Processed {oov_words.index((word, count)) + 1}/{len(oov_words)}...")
    
    print(f"\n✅ Dictionary saved to: {args.output}")
    print(f"   Total entries: {len(oov_words)}")
    
    # Print sample
    print(f"\nSample entries:")
    with open(args.output, 'r') as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            print(f"  {line.strip()}")


if __name__ == '__main__':
    main()
