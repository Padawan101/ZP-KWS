#!/usr/bin/env python3
"""
verify_mfa_alignment.py - Verify MFA alignment quality

"""

import argparse
from pathlib import Path
from collections import defaultdict
import random


def parse_args():
    parser = argparse.ArgumentParser(description='Verify MFA alignment')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input staging directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output alignment directory')
    parser.add_argument('--report', type=str, default='alignment_report.txt',
                        help='Output report file')
    parser.add_argument('--sample_size', type=int, default=10,
                        help='Number of TextGrids to sample for detailed check')
    return parser.parse_args()


def parse_textgrid(textgrid_path: Path) -> dict:
    """
    Parse TextGrid file to extract basic info
    Returns: {
        'num_tiers': int,
        'num_phones': int,
        'num_words': int,
        'duration': float
    }
    """
    info = {'num_tiers': 0, 'num_phones': 0, 'num_words': 0, 'duration': 0.0}
    
    try:
        with open(textgrid_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Count tiers
            info['num_tiers'] = content.count('item [')
            
            # Extract duration (xmax of File)
            for line in content.split('\n'):
                if 'xmax' in line and info['duration'] == 0.0:
                    try:
                        info['duration'] = float(line.split('=')[1].strip())
                        break
                    except:
                        pass
            
            # Count intervals containing phones/words
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'text = ' in line:
                    text = line.split('=')[1].strip().strip('"')
                    if text and text != '':
                        # Determine if it's word or phone tier based on previous lines
                        # Simple heuristic: if text has spaces, it's likely a word
                        if ' ' not in text and len(text) <= 3:
                            info['num_phones'] += 1
                        else:
                            info['num_words'] += 1
        
        return info
    except Exception as e:
        print(f"  Warning: Failed to parse {textgrid_path}: {e}")
        return None


def main():
    args = parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    
    print("=" * 70)
    print("MFA Alignment Verification")
    print("=" * 70)
    
    # Step 1: Count input files
    print("\n[Step 1] Counting input files...")
    wav_files = list(input_path.rglob("*.wav"))
    lab_files = list(input_path.rglob("*.lab"))
    print(f"  WAV files: {len(wav_files)}")
    print(f"  LAB files: {len(lab_files)}")
    
    # Step 2: Count output files
    print("\n[Step 2] Counting output TextGrids...")
    if not output_path.exists():
        print(f"  ERROR: Output directory {output_path} does not exist!")
        return
    
    textgrid_files = list(output_path.rglob("*.TextGrid"))
    print(f"  TextGrid files: {len(textgrid_files)}")
    
    # Step 3: Check alignment coverage
    print("\n[Step 3] Checking alignment coverage...")
    success_rate = len(textgrid_files) / len(wav_files) * 100 if wav_files else 0
    print(f"  Alignment success rate: {success_rate:.2f}%")
    
    if success_rate < 95:
        print(f"  ⚠️  Warning: Low success rate! Check for errors.")
    else:
        print(f"  ✅ Good alignment coverage")
    
    # Step 4: Find missing alignments
    print("\n[Step 4] Finding missing alignments...")
    wav_basenames = {f.stem for f in wav_files}
    textgrid_basenames = {f.stem for f in textgrid_files}
    missing = wav_basenames - textgrid_basenames
    
    print(f"  Missing alignments: {len(missing)}")
    if missing and len(missing) <= 20:
        print(f"  Missing files:")
        for m in sorted(list(missing)[:20]):
            print(f"    - {m}")
    
    # Step 5: Sample quality check
    print(f"\n[Step 5] Sampling {args.sample_size} TextGrids for quality check...")
    sample = random.sample(textgrid_files, min(args.sample_size, len(textgrid_files)))
    
    sample_stats = []
    for tg_file in sample:
        info = parse_textgrid(tg_file)
        if info:
            sample_stats.append(info)
            print(f"  {tg_file.name}: {info['num_phones']} phones, "
                  f"{info['num_words']} words, {info['duration']:.2f}s")
    
    # Step 6: Statistics
    print("\n[Step 6] Alignment statistics...")
    if sample_stats:
        avg_phones = sum(s['num_phones'] for s in sample_stats) / len(sample_stats)
        avg_words = sum(s['num_words'] for s in sample_stats) / len(sample_stats)
        avg_duration = sum(s['duration'] for s in sample_stats) / len(sample_stats)
        
        print(f"  Average phones per file: {avg_phones:.1f}")
        print(f"  Average words per file: {avg_words:.1f}")
        print(f"  Average duration: {avg_duration:.2f}s")
    
    # Step 7: Generate report
    print(f"\n[Step 7] Generating report to {args.report}...")
    with open(args.report, 'w', encoding='utf-8') as f:
        f.write("MFA Alignment Verification Report\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Input Directory: {input_path}\n")
        f.write(f"Output Directory: {output_path}\n\n")
        
        f.write(f"Input Files:\n")
        f.write(f"  WAV files: {len(wav_files)}\n")
        f.write(f"  LAB files: {len(lab_files)}\n\n")
        
        f.write(f"Output Files:\n")
        f.write(f"  TextGrid files: {len(textgrid_files)}\n")
        f.write(f"  Success rate: {success_rate:.2f}%\n\n")
        
        f.write(f"Missing Alignments ({len(missing)}):\n")
        for m in sorted(missing):
            f.write(f"  {m}\n")
        
        if sample_stats:
            f.write(f"\nSample Statistics (n={len(sample_stats)}):\n")
            f.write(f"  Average phones: {avg_phones:.1f}\n")
            f.write(f"  Average words: {avg_words:.1f}\n")
            f.write(f"  Average duration: {avg_duration:.2f}s\n")
    
    print(f"\n✅ Verification complete! Report saved to {args.report}")
    
    # Final verdict
    print("\n" + "=" * 70)
    if success_rate >= 95 and len(missing) < len(wav_files) * 0.05:
        print("✅ PASS: Alignment quality looks good!")
    elif success_rate >= 80:
        print("⚠️  WARNING: Alignment partially successful, review missing files")
    else:
        print("❌ FAIL: Low alignment success rate, check MFA logs")


if __name__ == '__main__':
    main()
