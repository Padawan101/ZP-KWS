#!/usr/bin/env python3
"""
build_ipa_mapping.py

掃描所有 TextGrid 收集 MFA 輸出的 IPA 音素符號，建立 IPA → ARPAbet (去 stress) 映射表。
根據 MFA_AuxCE_Design.md Step 3 設計。
"""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path


# 預定義的 IPA → ARPAbet (去 stress) 映射表
# 基於 MFA english_mfa acoustic model 的輸出
# 包含所有 78 個在 365K TextGrid 檔案中發現的 IPA 符號
IPA_TO_ARPABET = {
    # ===== 母音 (Vowels) =====
    "ɑ": "AA", "ɑː": "AA",
    "æ": "AE",
    "ʌ": "AH", "ɐ": "AH",  # MFA 常用 ɐ 代替 ʌ
    "ɔ": "AO", "ɒ": "AO", "ɔː": "AO", "ɒː": "AO",  # added ɒː
    "aʊ": "AW", "aw": "AW",
    "aɪ": "AY", "aj": "AY",
    "ɛ": "EH", "e": "EH",
    "ɝ": "ER", "ɚ": "ER", "ɜː": "ER", "ɹ̩": "ER",
    "eɪ": "EY", "ej": "EY",
    "ɪ": "IH",
    "i": "IY", "iː": "IY",
    "oʊ": "OW", "ow": "OW", "əʊ": "OW", "o": "OW",
    "ɔɪ": "OY", "oj": "OY", "ɔj": "OY",  # added ɔj
    "ʊ": "UH",
    "u": "UW", "uː": "UW",
    "ʉ": "UW", "ʉː": "UW",  # close central rounded vowel → UW
    "ə": "AH",  # schwa → AH (unstressed vowel)
    
    # ===== 子音 (Consonants) =====
    "b": "B",
    "bʲ": "B",  # palatalized b
    "tʃ": "CH",
    "d": "D",
    "d̪": "D",  # dental d (很常見: 68,705 次)
    "dʲ": "D",  # palatalized d
    "ð": "DH",
    "f": "F",
    "fʲ": "F",  # palatalized f
    "ɡ": "G", "g": "G",
    "ɡʷ": "G",  # labialized g
    "h": "HH",
    "dʒ": "JH",
    "ɟ": "JH",  # voiced palatal stop → JH (closest English sound)
    "ɟʷ": "JH",  # labialized voiced palatal stop
    "k": "K",
    "kʰ": "K",  # aspirated k (很常見: 24,151 次)
    "kʷ": "K",  # labialized k
    "c": "K",   # voiceless palatal stop → K
    "cʰ": "K",  # aspirated palatal stop
    "cʷ": "K",  # labialized palatal stop
    "l": "L", "ɫ": "L",  # dark L
    "ɫ̩": "L",  # syllabic dark L
    "ʎ": "L",   # palatal lateral approximant → L
    "m": "M",
    "mʲ": "M",  # palatalized m
    "m̩": "M",   # syllabic m
    "n": "N",
    "n̩": "N",   # syllabic n
    "ɲ": "N",   # palatal nasal → N (closest)
    "ŋ": "NG",
    "p": "P",
    "pʰ": "P",  # aspirated p (很常見: 22,666 次)
    "pʲ": "P",  # palatalized p
    "pʷ": "P",  # labialized p
    "ɹ": "R", "r": "R",
    "ɾ": "R",   # alveolar tap → R (American English flap)
    "ɾʲ": "R",  # palatalized tap
    "ɾ̃": "R",   # nasalized tap
    "s": "S",
    "ʃ": "SH",
    "t": "T",
    "t̪": "T",   # dental t
    "tʰ": "T",  # aspirated t (很常見: 16,334 次)
    "tʲ": "T",  # palatalized t (很常見: 31,742 次)
    "tʷ": "T",  # labialized t
    "θ": "TH",
    "v": "V",
    "vʲ": "V",  # palatalized v
    "w": "W",
    "j": "Y",
    "ç": "Y",   # voiceless palatal fricative → Y (closest, often allophone of /hj/)
    "z": "Z",
    "ʒ": "ZH",
    "ʔ": "SIL",  # glottal stop → treat as silence boundary
    
    # ===== 特殊符號 =====
    "spn": "SPN",  # spoken noise
    "sil": "SIL",  # silence
    "": "SIL",     # empty interval
}



def parse_textgrid_phones(textgrid_path: str) -> list:
    """
    解析 TextGrid 檔案的 phones tier，提取所有 IPA 符號。
    
    Args:
        textgrid_path: TextGrid 檔案路徑
        
    Returns:
        list of IPA phones found in this file
    """
    phones = []
    
    try:
        with open(textgrid_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try with latin-1 encoding as fallback
        with open(textgrid_path, 'r', encoding='latin-1') as f:
            content = f.read()
    
    # Find the phones tier section
    # Looking for: name = "phones" followed by intervals
    lines = content.split('\n')
    in_phones_tier = False
    
    for i, line in enumerate(lines):
        # Check if we're entering the phones tier
        if 'name = "phones"' in line:
            in_phones_tier = True
            continue
        
        # Check if we're leaving the phones tier (entering another tier)
        if in_phones_tier and 'name = "words"' in line:
            break
            
        # Extract phone text within phones tier
        if in_phones_tier and 'text = "' in line:
            # Extract text between quotes
            match = re.search(r'text = "([^"]*)"', line)
            if match:
                phone = match.group(1).strip()
                phones.append(phone)
    
    return phones


def scan_all_textgrids(textgrid_dir: str) -> Counter:
    """
    掃描目錄下所有 TextGrid 檔案，統計所有 IPA 音素。
    
    Args:
        textgrid_dir: TextGrid 目錄路徑
        
    Returns:
        Counter of IPA phones
    """
    phone_counter = Counter()
    textgrid_count = 0
    error_count = 0
    
    textgrid_dir = Path(textgrid_dir)
    
    # Find all .TextGrid files recursively
    textgrid_files = list(textgrid_dir.glob("**/*.TextGrid"))
    print(f"Found {len(textgrid_files)} TextGrid files")
    
    for tg_path in textgrid_files:
        try:
            phones = parse_textgrid_phones(str(tg_path))
            phone_counter.update(phones)
            textgrid_count += 1
            
            if textgrid_count % 10000 == 0:
                print(f"Processed {textgrid_count} TextGrid files...")
                
        except Exception as e:
            error_count += 1
            if error_count <= 10:  # Only print first 10 errors
                print(f"Error parsing {tg_path}: {e}")
    
    print(f"\nProcessed {textgrid_count} TextGrid files with {error_count} errors")
    return phone_counter


def build_mapping(phone_counter: Counter) -> dict:
    """
    根據收集的 IPA 音素建立完整映射表。
    
    Returns:
        dict with:
            - ipa_to_arpabet: IPA → ARPAbet mapping
            - arpabet_vocab: phoneme vocabulary (PAD, SIL, SPN, + 39 phonemes)
            - unmapped: list of IPA symbols without mapping
    """
    # Start with predefined mapping
    ipa_to_arpabet = dict(IPA_TO_ARPABET)
    
    # Find unmapped phones
    unmapped = []
    for phone in phone_counter:
        if phone not in ipa_to_arpabet:
            unmapped.append((phone, phone_counter[phone]))
    
    # Sort unmapped by count (descending)
    unmapped.sort(key=lambda x: -x[1])
    
    # Build phoneme vocabulary (42 classes)
    arpabet_vocab = {
        "<PAD>": 0,  # padding，不參與 loss 計算
        "SIL": 1,    # silence / empty
        "SPN": 2,    # spoken noise
        # 39 基礎 ARPAbet (按字母順序)
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
    
    return {
        "ipa_to_arpabet": ipa_to_arpabet,
        "arpabet_vocab": arpabet_vocab,
        "n_phonemes": len(arpabet_vocab),
        "unmapped": unmapped,
        "phone_counts": dict(phone_counter.most_common()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build IPA → ARPAbet mapping from MFA TextGrid files"
    )
    parser.add_argument(
        "--textgrid_dir",
        type=str,
        required=True,
        help="Directory containing TextGrid files (recursive search)"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed statistics"
    )
    args = parser.parse_args()
    
    print(f"Scanning TextGrid files in: {args.textgrid_dir}")
    phone_counter = scan_all_textgrids(args.textgrid_dir)
    
    print(f"\nTotal unique IPA phones found: {len(phone_counter)}")
    
    if args.verbose:
        print("\nTop 50 phones by frequency:")
        for phone, count in phone_counter.most_common(50):
            arpabet = IPA_TO_ARPABET.get(phone, "???")
            print(f"  '{phone}' ({arpabet}): {count:,}")
    
    # Build mapping
    mapping = build_mapping(phone_counter)
    
    # Report unmapped phones
    if mapping["unmapped"]:
        print(f"\n⚠️  Warning: {len(mapping['unmapped'])} unmapped IPA phones found:")
        for phone, count in mapping["unmapped"][:20]:
            print(f"  '{phone}': {count:,} occurrences")
        if len(mapping["unmapped"]) > 20:
            print(f"  ... and {len(mapping['unmapped']) - 20} more")
    else:
        print("\n✓ All IPA phones have valid mappings!")
    
    # Save to JSON
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Mapping saved to: {args.output_json}")
    print(f"  - n_phonemes: {mapping['n_phonemes']}")
    print(f"  - Total IPA phones mapped: {len(mapping['ipa_to_arpabet'])}")


if __name__ == "__main__":
    main()
