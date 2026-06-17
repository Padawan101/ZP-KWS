"""
Standalone Evaluation Script for P-PhonMatchNet Checkpoints

Loads a trained checkpoint and runs the EXACT same validation pipeline
as train_personalized.py, producing:
  1. C-KWS / TB-KWS / TO-KWS / SV metrics (EER, AUC, FRR@FAR)
  2. Score diagnostics (if --diagnostics flag is set)
  3. P_phon lightweight stats

Usage:
    # Evaluate on LibriPhrase (Easy + Hard)
    python eval_checkpoint.py --checkpoint results/aux_biGRU/best_checkpoint.pth

    # Evaluate on all datasets (LibriPhrase + GSC + Qualcomm)
    python eval_checkpoint.py --checkpoint results/aux_biGRU/best_checkpoint.pth --eval_gsc_qualcomm

    # With score diagnostics
    python eval_checkpoint.py --checkpoint results/aux_biGRU/best_checkpoint.pth --diagnostics

    # Override batch size / workers
    python eval_checkpoint.py --checkpoint results/aux_biGRU/best_checkpoint.pth --batch_size 512 --num_workers 4
"""

import argparse
import os
import sys
import warnings
import gc
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# === Suppress warnings ===
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import nltk
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('punkt', quiet=True)

import numpy as np
import torch
import torch.nn.functional as F

from model import p_ukws
from dataset import personalized_libriphrase
from dataset import KWSDataLoader
from dataset.personalized_google import PersonalizedGoogleCommandsDataset
from dataset.personalized_qualcomm import PersonalizedQualcommDataset
from criterion.personalized_metrics import PersonalizedKWSMetrics

seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a P-PhonMatchNet checkpoint"
    )
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='Path to checkpoint .pth file (e.g. best_checkpoint.pth)')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Override batch size (default: use checkpoint args)')
    parser.add_argument('--num_workers', type=int, default=10,
                        help='Override num_workers (default: use checkpoint args)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    # Dataset overrides
    parser.add_argument('--train_pkl', type=str, default="/padawan/train_combined.pkl",
                        help='Override training pkl (for vocab)')
    parser.add_argument('--libriphrase_pkl', type=str, default="/padawan/test_500h.pkl",
                        help='Override LibriPhrase test pkl')
    parser.add_argument('--google_pkl', type=str, default="/padawan/google_speech_commands/google.pkl",
                        help='Override Google test pkl')
    parser.add_argument('--qualcomm_pkl', type=str, default="/padawan/qualcomm/qualcomm.pkl",
                        help='Override Qualcomm test pkl')

    # Evaluation scope
    parser.add_argument('--eval_gsc_qualcomm', action='store_true', default=True,
                        help='Also evaluate on GSC and Qualcomm datasets')
    parser.add_argument('--fusion_mode', type=str, default=None, choices=['multiply', 'harmonic', 'min'],
                        help='Override fusion mode (default: use checkpoint args)')

    # Calibration override (no retraining needed)
    parser.add_argument('--spk_scale', type=float, default=None,
                        help='Override spk_scale for P_spk = sigmoid(scale*cos + bias)')
    parser.add_argument('--spk_bias', type=float, default=None,
                        help='Override spk_bias for P_spk = sigmoid(scale*cos + bias)')

    # Grid search over (scale, bias) — post-hoc, no retraining
    parser.add_argument('--grid_search', action='store_true', default=False,
                        help='Run grid search over (spk_scale, spk_bias) combinations')
    parser.add_argument('--grid_scales', type=str, default='3,5,7,10,15',
                        help='Comma-separated scale values for grid search')
    parser.add_argument('--grid_biases', type=str, default='-7,-6,-5,-4,-3,-2',
                        help='Comma-separated bias values for grid search')

    # Diagnostics
    parser.add_argument('--diagnostics', action='store_true', default=False,
                        help='Run score diagnostics (detailed per-category analysis)')
    parser.add_argument('--save_raw_scores', action='store_true', default=False,
                        help='Save raw scores as .npz for later analysis')

    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same dir as checkpoint)')

    return parser.parse_args()


def load_checkpoint(ckpt_path, device):
    """Load checkpoint and return model state_dict and saved args."""
    print(f"\n>> Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    saved_args = checkpoint['args']  # dict
    epoch = checkpoint.get('epoch', -1)
    best_val_eer = checkpoint.get('best_val_eer', None)

    print(f">> Checkpoint epoch: {epoch}")
    if best_val_eer is not None:
        print(f">> Best val EER: {best_val_eer*100:.2f}%")

    # Print key config
    print(f">> Config: mode={saved_args.get('mode')}, "
          f"audio_input={saved_args.get('audio_input')}, "
          f"text_input={saved_args.get('text_input')}, "
          f"personalized={saved_args.get('personalized')}")
    print(f">>         bidirectional={saved_args.get('bidirectional', False)}, "
          f"gru_layers={saved_args.get('gru_layers', 2)}, "
          f"disable_hybrid_encoder={saved_args.get('disable_hybrid_encoder', False)}")

    return checkpoint, saved_args


def build_model(saved_args, device):
    """Build P_UKWS model from saved args (same logic as train_personalized.py)."""
    vocab = saved_args.get('vocab', 42)

    kwargs = {
        'vocab': vocab,
        'text_input': saved_args.get('text_input', 'g2p_embed'),
        'audio_input': saved_args.get('audio_input', 'both'),
        'stack_extractor': saved_args.get('stack_extractor', False),
        'frame_length': saved_args.get('frame_length', 400),
        'hop_length': saved_args.get('hop_length', 160),
        'num_mel': 40,
        'sample_rate': saved_args.get('sample_rate', 16000),
        'log_mel': saved_args.get('log_mel', False),
        'mode': saved_args.get('mode', 'TO-KWS'),
        'speaker_encoder_path': saved_args.get('speaker_encoder_path', 'model/speaker/efficient_tdnn'),
        # Ablation
        'disable_film': saved_args.get('disable_film', False),
        'disable_sv_branch': saved_args.get('disable_sv_branch', False),
        'freeze_speaker_encoder': saved_args.get('freeze_speaker_encoder', True),
        'finetuned_speaker_encoder_path': saved_args.get('finetuned_speaker_encoder', None),
        'film_target': saved_args.get('film_target', 'fused'),
        'film_gate_type': saved_args.get('film_gate_type', 'pspk'),
        # MFA Auxiliary CE
        'enable_aux_ce': saved_args.get('enable_aux_ce', False),
        'n_phonemes': saved_args.get('n_phonemes', 42),
        # A5/A6
        'gemb_drop_rate': saved_args.get('gemb_drop_rate', 0.0),
        'gemb_curriculum': saved_args.get('gemb_curriculum', False),
        'gemb_warmup_epochs': saved_args.get('gemb_warmup_epochs', 5),
        'gemb_ramp_epochs': saved_args.get('gemb_ramp_epochs', 10),
        # Calibration
        'calibration_mode': saved_args.get('calibration_mode', 'full'),
        # Fusion
        'fusion_mode': saved_args.get('fusion_mode', 'multiply'),
        # Audio encoder
        'disable_hybrid_encoder': saved_args.get('disable_hybrid_encoder', False),
        'disable_ldn_norm': saved_args.get('disable_ldn_norm', False),
        'gru_layers': saved_args.get('gru_layers', 2),
        'bidirectional': saved_args.get('bidirectional', False),
        # Stream fusion
        'stream_fusion': saved_args.get('stream_fusion', 'add'),
    }

    model = p_ukws.P_UKWS(**kwargs)
    model.to(device)
    return model


def create_eval_loaders(saved_args, cli_args):
    """Create evaluation dataloaders (same as train_personalized.py)."""
    # Build a namespace object that matches what prepare_loader expects
    class ArgsProxy:
        pass

    args = ArgsProxy()
    # Copy all saved args
    for k, v in saved_args.items():
        setattr(args, k, v)

    # Apply CLI overrides
    if cli_args.batch_size is not None:
        args.batch_size = cli_args.batch_size
    if cli_args.num_workers is not None:
        args.num_workers = cli_args.num_workers
    if cli_args.train_pkl is not None:
        args.train_pkl = cli_args.train_pkl
    if cli_args.libriphrase_pkl is not None:
        args.libriphrase_pkl = cli_args.libriphrase_pkl

    # Ensure required defaults
    if not hasattr(args, 'batch_size'):
        args.batch_size = 64
    if not hasattr(args, 'num_workers'):
        args.num_workers = 2
    if not hasattr(args, 'train_pkl'):
        args.train_pkl = '/padawan/train_combined.pkl'
    if not hasattr(args, 'libriphrase_pkl'):
        args.libriphrase_pkl = '/padawan/test_500h.pkl'
    if not hasattr(args, 'speaker_ratio'):
        args.speaker_ratio = 0.5
    if not hasattr(args, 'max_train_samples'):
        args.max_train_samples = None
    if not hasattr(args, 'max_val_samples'):
        args.max_val_samples = None
    if not hasattr(args, 'personalized'):
        args.personalized = True
    if not hasattr(args, 'frame_labels_path'):
        args.frame_labels_path = None

    # Create dataloaders (reuse the exact same function)
    _, eval_loaders, vocab, _ = personalized_libriphrase.create_personalized_dataloaders(
        args,
        train_personalized=args.personalized
    )

    return eval_loaders, vocab, args


def create_extended_loaders(saved_args, cli_args):
    """Create GSC and Qualcomm loaders."""
    batch_size = cli_args.batch_size or saved_args.get('batch_size', 64)
    num_workers = cli_args.num_workers or saved_args.get('num_workers', 2)
    audio_input = saved_args.get('audio_input', 'raw')
    text_input = saved_args.get('text_input', 'g2p_embed')
    frame_length = saved_args.get('frame_length', 400)
    hop_length = saved_args.get('hop_length', 160)
    personalized = saved_args.get('personalized', True)
    speaker_ratio = saved_args.get('speaker_ratio', 0.5)

    gemb_dir = None if audio_input == "raw" else '/padawan/google_speech_embedding/DB'

    google_pkl = cli_args.google_pkl or saved_args.get('google_pkl', '/padawan/google_speech_commands/google.pkl')
    qualcomm_pkl = cli_args.qualcomm_pkl or saved_args.get('qualcomm_pkl', '/padawan/qualcomm/qualcomm.pkl')

    loaders = {}

    # GSC
    try:
        print(">> Creating Google Speech Commands test loader...")
        gsc = PersonalizedGoogleCommandsDataset(
            batch_size=batch_size,
            gemb_dir=gemb_dir,
            features=text_input,
            testset_only=False,
            shuffle=False,
            pkl=google_pkl,
            frame_length=frame_length,
            hop_length=hop_length,
            personalized=personalized,
            speaker_ratio=speaker_ratio,
        )
        loaders['google_speech_commands'] = KWSDataLoader(
            gsc, batch_size,
            shuffle=True, pin_memory=True, drop_last=True,
            num_workers=num_workers,
        )
        print(f"   GSC: {len(gsc)} samples")
    except Exception as e:
        print(f"   [WARNING] Failed to create GSC loader: {e}")
        loaders['google_speech_commands'] = None

    # Qualcomm
    try:
        print(">> Creating Qualcomm test loader...")
        qualcomm = PersonalizedQualcommDataset(
            batch_size=batch_size,
            gemb_dir=gemb_dir,
            features=text_input,
            shuffle=False,
            pkl=qualcomm_pkl,
            frame_length=frame_length,
            hop_length=hop_length,
            personalized=personalized,
            speaker_ratio=speaker_ratio,
        )
        loaders['qualcomm'] = KWSDataLoader(
            qualcomm, batch_size,
            shuffle=True, pin_memory=True, drop_last=True,
            num_workers=num_workers,
        )
        print(f"   Qualcomm: {len(qualcomm)} samples")
    except Exception as e:
        print(f"   [WARNING] Failed to create Qualcomm loader: {e}")
        loaders['qualcomm'] = None

    return loaders


def evaluate_loader(model, loader, loader_name, saved_args, device, collect_raw=False):
    """
    Run evaluation on a single loader.
    Replicates the exact validation loop from train_personalized.py.
    
    If collect_raw=True, also returns raw scores for post-hoc grid search.
    """
    audio_input = saved_args.get('audio_input', 'both')
    personalized = saved_args.get('personalized', True)
    fusion_mode = saved_args.get('fusion_mode', 'multiply')

    metrics_calculator = PersonalizedKWSMetrics(fusion_mode=fusion_mode)

    # P_phon stats accumulators
    pphon_pos_means, pphon_pos_mins = [], []
    pphon_neg_means, pphon_neg_mins = [], []

    # Raw data collection for grid search
    raw_p_utt_list = []
    raw_p_spk_list = []
    raw_kw_label_list = []
    raw_spk_label_list = []
    raw_category_list = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Eval {loader_name}",
                                                 dynamic_ncols=True, leave=False)):
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # Prepare inputs (same as train_personalized.py validation)
            if audio_input == "raw":
                speech_input = batch["x"]
                speech_len = batch["x_len"]
            elif audio_input == "google_embed":
                speech_input = batch["gemb"]
                speech_len = batch["gemb_len"]
            elif audio_input in ("both", "enhanced_gembed"):
                speech_input = (batch["x"], batch["gemb"])
                speech_len = (batch["x_len"], batch["gemb_len"])

            # Forward pass
            if personalized:
                output = model(
                    speech=speech_input,
                    text=batch["y"],
                    speech_len=speech_len,
                    text_len=batch["y_len"],
                    enrollment_audio=batch["enrollment_audio"],
                    raw_audio_for_spk=batch.get("x")
                )
            else:
                output = model(
                    speech=speech_input,
                    text=batch["y"],
                    speech_len=speech_len,
                    text_len=batch["y_len"],
                )

            # Collect scores
            keyword_scores = output['P_utt']
            if personalized:
                speaker_scores = output.get('P_spk', torch.ones_like(keyword_scores))
            else:
                speaker_scores = torch.ones_like(keyword_scores)

            # Collect labels
            if personalized:
                keyword_labels = batch['z_keyword']
                speaker_labels = batch.get('speaker_label', torch.zeros_like(keyword_labels))
                categories = batch.get('category', ['ts-tk'] * len(keyword_labels))
            else:
                keyword_labels = batch['z']
                speaker_labels = torch.zeros_like(keyword_labels)
                categories = ['ts-tk'] * len(keyword_labels)

            # Update metrics
            metrics_calculator.update(
                keyword_scores=keyword_scores,
                speaker_scores=speaker_scores,
                keyword_labels=keyword_labels,
                speaker_labels=speaker_labels,
                categories=categories
            )

            # Collect raw data for grid search
            if collect_raw:
                raw_p_utt_list.append(keyword_scores.cpu())
                raw_p_spk_list.append(speaker_scores.cpu())
                raw_kw_label_list.append(keyword_labels.cpu())
                raw_spk_label_list.append(speaker_labels.cpu())
                raw_category_list.extend(
                    categories if isinstance(categories, list) else categories.tolist()
                )

            # P_phon lightweight stats
            if 'seq_ce_logit' in output:
                seq_logit = output['seq_ce_logit']
                seq_mask = output.get('seq_ce_logit_mask', torch.ones_like(seq_logit))
                P_phon = torch.sigmoid(seq_logit)
                P_phon_masked = P_phon * seq_mask
                valid_counts = seq_mask.sum(dim=-1).clamp(min=1)
                pphon_mean = P_phon_masked.sum(dim=-1) / valid_counts

                P_phon_for_min = P_phon.clone()
                P_phon_for_min[seq_mask == 0] = 1.0
                pphon_min = P_phon_for_min.min(dim=-1).values

                z_kw = keyword_labels.squeeze(-1) if keyword_labels.dim() > 1 else keyword_labels
                pos_mask = z_kw == 1
                neg_mask = z_kw == 0

                if pos_mask.any():
                    pphon_pos_means.append(pphon_mean[pos_mask].cpu())
                    pphon_pos_mins.append(pphon_min[pos_mask].cpu())
                if neg_mask.any():
                    pphon_neg_means.append(pphon_mean[neg_mask].cpu())
                    pphon_neg_mins.append(pphon_min[neg_mask].cpu())

    # Compute final metrics
    all_mode_metrics = metrics_calculator.compute()

    # P_phon summary
    pphon_stats = None
    if pphon_pos_means or pphon_neg_means:
        pos_mean_all = torch.cat(pphon_pos_means).mean().item() if pphon_pos_means else 0
        pos_min_all = torch.cat(pphon_pos_mins).mean().item() if pphon_pos_mins else 0
        neg_mean_all = torch.cat(pphon_neg_means).mean().item() if pphon_neg_means else 0
        neg_min_all = torch.cat(pphon_neg_mins).mean().item() if pphon_neg_mins else 0
        pphon_stats = {
            'positive_mean': pos_mean_all,
            'positive_min': pos_min_all,
            'negative_mean': neg_mean_all,
            'negative_min': neg_min_all,
            'gap_mean': pos_mean_all - neg_mean_all,
            'gap_min': pos_min_all - neg_min_all,
        }

    # Assemble raw data dict
    raw_data = None
    if collect_raw and raw_p_utt_list:
        raw_data = {
            'P_utt': torch.cat(raw_p_utt_list).numpy(),
            'P_spk': torch.cat(raw_p_spk_list).numpy(),
            'keyword_labels': torch.cat(raw_kw_label_list).numpy(),
            'speaker_labels': torch.cat(raw_spk_label_list).numpy(),
            'categories': raw_category_list,
        }

    return all_mode_metrics, metrics_calculator, pphon_stats, raw_data


def print_metrics(loader_name, metrics, pphon_stats=None):
    """Pretty-print evaluation results (same format as train_personalized.py)."""
    print(f"\n[{loader_name}]")
    for mode in ['C-KWS', 'TB-KWS', 'TO-KWS']:
        if mode in metrics and 'error' not in metrics[mode]:
            m = metrics[mode]
            print(
                f"  {mode:8s} - EER: {m['EER']*100:6.2f}%  "
                f"AUC: {m['AUC']*100:6.2f}%  "
                f"FRR@1%: {m.get('FRR@FAR1%', 0)*100:6.2f}%  "
                f"FRR@5%: {m.get('FRR@FAR5%', 0)*100:6.2f}%  "
                f"FRR@10%: {m.get('FRR@FAR10%', 0)*100:6.2f}%"
            )
    # SV metrics
    if 'SV' in metrics and 'error' not in metrics['SV']:
        m = metrics['SV']
        print(
            f"  {'SV':8s} - EER: {m['EER']*100:6.2f}%  "
            f"AUC: {m['AUC']*100:6.2f}%"
        )

    if pphon_stats:
        print(f"  P_phon: pos_mean={pphon_stats['positive_mean']:.3f}, "
              f"neg_mean={pphon_stats['negative_mean']:.3f}, "
              f"gap={pphon_stats['gap_mean']:.3f} | "
              f"pos_min={pphon_stats['positive_min']:.3f}, "
              f"neg_min={pphon_stats['negative_min']:.3f}, "
              f"gap={pphon_stats['gap_min']:.3f}")


def back_compute_cos_sim(p_spk, spk_scale, spk_bias):
    """
    Back-compute cosine similarity from P_spk.
    P_spk = sigmoid(scale * cos + bias)
    => logit(P_spk) = scale * cos + bias
    => cos = (logit(P_spk) - bias) / scale
    """
    p_clamped = np.clip(p_spk, 1e-7, 1 - 1e-7)
    logit = np.log(p_clamped / (1 - p_clamped))
    cos_sim = (logit - spk_bias) / spk_scale
    return np.clip(cos_sim, -1, 1)


def compute_metrics_for_calibration(raw_data, scale, bias, fusion_mode='multiply'):
    """
    Recompute TO-KWS/TB-KWS/C-KWS metrics for a given (scale, bias).
    Uses collected raw data — no forward pass needed.
    """
    cos_sim = raw_data['cos_sim']
    P_utt = raw_data['P_utt']
    keyword_labels = raw_data['keyword_labels']
    speaker_labels = raw_data['speaker_labels']
    categories = raw_data['categories']

    # Recompute P_spk with new calibration
    P_spk = 1.0 / (1.0 + np.exp(-(scale * cos_sim + bias)))

    # Convert to tensors for PersonalizedKWSMetrics
    P_utt_t = torch.from_numpy(P_utt).float()
    P_spk_t = torch.from_numpy(P_spk).float()
    kw_labels_t = torch.from_numpy(keyword_labels).float()
    spk_labels_t = torch.from_numpy(speaker_labels).float()

    calc = PersonalizedKWSMetrics(fusion_mode=fusion_mode)
    calc.update(
        keyword_scores=P_utt_t,
        speaker_scores=P_spk_t,
        keyword_labels=kw_labels_t,
        speaker_labels=spk_labels_t,
        categories=categories,
    )
    return calc.compute()


def run_grid_search(all_raw_data, scales, biases, fusion_mode, output_dir):
    """
    Grid search over (scale, bias) combinations across all datasets.
    
    Args:
        all_raw_data: dict of {dataset_name: raw_data_dict}
        scales: list of float
        biases: list of float
        fusion_mode: str
        output_dir: str
    """
    print("\n" + "=" * 80)
    print("Grid Search: Calibration (scale, bias)")
    print("=" * 80)

    # Collect results
    grid_results = {}  # {dataset_name: {(scale, bias): metrics_dict}}

    for ds_name, raw_data in all_raw_data.items():
        if raw_data is None:
            continue
        print(f"\n>> Searching {ds_name}...")
        grid_results[ds_name] = {}

        for s in scales:
            for b in biases:
                metrics = compute_metrics_for_calibration(raw_data, s, b, fusion_mode)
                grid_results[ds_name][(s, b)] = metrics

    # === Print results ===
    # For each metric of interest, find the best (scale, bias)
    target_modes = ['C-KWS', 'TB-KWS', 'TO-KWS']
    target_metrics = ['EER', 'FRR@FAR1%', 'FRR@FAR5%']

    for ds_name, results in grid_results.items():
        print(f"\n{'=' * 80}")
        print(f"  {ds_name}")
        print(f"{'=' * 80}")

        # Print TO-KWS EER table (most important)
        print(f"\n  TO-KWS EER (%) — rows=scale, cols=bias:")
        print(f"  {'scale↓ bias→':>12s}", end="")
        for b in biases:
            print(f"  {b:>7.1f}", end="")
        print()
        for s in scales:
            print(f"  {s:>12.1f}", end="")
            for b in biases:
                m = results.get((s, b), {})
                to_kws = m.get('TO-KWS', {})
                eer = to_kws.get('EER', float('nan'))
                print(f"  {eer*100:>7.2f}", end="")
            print()

        # Print TB-KWS EER table
        print(f"\n  TB-KWS EER (%):")
        print(f"  {'scale↓ bias→':>12s}", end="")
        for b in biases:
            print(f"  {b:>7.1f}", end="")
        print()
        for s in scales:
            print(f"  {s:>12.1f}", end="")
            for b in biases:
                m = results.get((s, b), {})
                tb_kws = m.get('TB-KWS', {})
                eer = tb_kws.get('EER', float('nan'))
                print(f"  {eer*100:>7.2f}", end="")
            print()

        # Print TO-KWS FRR@FAR1% table
        print(f"\n  TO-KWS FRR@FAR1% (%):")
        print(f"  {'scale↓ bias→':>12s}", end="")
        for b in biases:
            print(f"  {b:>7.1f}", end="")
        print()
        for s in scales:
            print(f"  {s:>12.1f}", end="")
            for b in biases:
                m = results.get((s, b), {})
                to_kws = m.get('TO-KWS', {})
                frr = to_kws.get('FRR@FAR1%', float('nan'))
                print(f"  {frr*100:>7.2f}", end="")
            print()

        # Find best combos
        print(f"\n  Best (scale, bias) per metric:")
        for mode in target_modes:
            for metric_name in target_metrics:
                best_val = float('inf')
                best_sb = None
                for (s, b), m in results.items():
                    if mode in m and metric_name in m[mode]:
                        v = m[mode][metric_name]
                        if v < best_val:
                            best_val = v
                            best_sb = (s, b)
                if best_sb is not None:
                    print(f"    {mode:8s} {metric_name:12s}: {best_val*100:6.2f}% @ scale={best_sb[0]:.1f}, bias={best_sb[1]:.1f}")

    # Save full results to file
    report_path = os.path.join(output_dir, "grid_search_results.txt")
    with open(report_path, 'w') as f:
        f.write("Grid Search Results: Calibration (scale, bias)\n")
        f.write(f"Scales: {scales}\n")
        f.write(f"Biases: {biases}\n")
        f.write(f"Fusion: {fusion_mode}\n")
        f.write("=" * 80 + "\n\n")

        for ds_name, results in grid_results.items():
            f.write(f"[{ds_name}]\n\n")

            for mode in target_modes:
                for metric_name in ['EER', 'AUC', 'FRR@FAR1%', 'FRR@FAR5%', 'FRR@FAR10%']:
                    f.write(f"  {mode} {metric_name} (%):\n")
                    f.write(f"  {'s \\ b':>8s}")
                    for b in biases:
                        f.write(f"  {b:>7.1f}")
                    f.write("\n")
                    for s in scales:
                        f.write(f"  {s:>8.1f}")
                        for b in biases:
                            m = results.get((s, b), {})
                            mode_m = m.get(mode, {})
                            v = mode_m.get(metric_name, float('nan'))
                            if metric_name == 'AUC':
                                f.write(f"  {v*100:>7.2f}")
                            else:
                                f.write(f"  {v*100:>7.2f}")
                        f.write("\n")
                    f.write("\n")

            # SV metrics (constant across calibration — sanity check)
            sample_m = list(results.values())[0] if results else {}
            sv_m = sample_m.get('SV', {})
            if sv_m:
                f.write(f"  SV EER: {sv_m.get('EER', 0)*100:.2f}% (constant — does not depend on calibration)\n")
            f.write("\n")

    print(f"\n>> Grid search report saved to: {report_path}")
    return grid_results


def main():
    args = parse_args()

    # Determine device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print(">> CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    # Load checkpoint
    checkpoint, saved_args = load_checkpoint(args.checkpoint, device)

    # Override fusion mode if provided
    if args.fusion_mode is not None:
        print(f">> Overriding fusion_mode from {saved_args.get('fusion_mode', 'multiply')} to {args.fusion_mode}")
        saved_args['fusion_mode'] = args.fusion_mode

    # Determine output dir
    output_dir = args.output_dir or str(Path(args.checkpoint).parent)

    # Build model
    print("\n>> Building model...")
    model = build_model(saved_args, device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # === Calibration override ===
    orig_scale = None
    orig_bias = None
    if hasattr(model, 'spk_scale'):
        orig_scale = model.spk_scale.data.item() if isinstance(model.spk_scale, torch.nn.Parameter) else model.spk_scale.item()
    if hasattr(model, 'spk_bias'):
        orig_bias = model.spk_bias.data.item() if isinstance(model.spk_bias, torch.nn.Parameter) else model.spk_bias.item()

    if args.spk_scale is not None or args.spk_bias is not None:
        new_scale = args.spk_scale if args.spk_scale is not None else orig_scale
        new_bias = args.spk_bias if args.spk_bias is not None else orig_bias
        print(f"\n>> Overriding calibration: scale={orig_scale} → {new_scale}, bias={orig_bias} → {new_bias}")
        if hasattr(model, 'spk_scale'):
            if isinstance(model.spk_scale, torch.nn.Parameter):
                model.spk_scale.data.fill_(new_scale)
            else:
                model.spk_scale.fill_(new_scale)
        if hasattr(model, 'spk_bias'):
            if isinstance(model.spk_bias, torch.nn.Parameter):
                model.spk_bias.data.fill_(new_bias)
            else:
                model.spk_bias.fill_(new_bias)

    # Count parameters
    param_counts = model.count_parameters()
    print(f">> Parameters: total={param_counts['total']:,}, "
          f"trainable={param_counts['trainable']:,}, "
          f"frozen={param_counts['frozen']:,}")

    # Whether we need raw data for grid search
    collect_raw = args.grid_search
    all_raw_data = {}

    # Create LibriPhrase eval loaders
    print("\n>> Creating LibriPhrase evaluation loaders...")
    eval_loaders, vocab, proxy_args = create_eval_loaders(saved_args, args)

    # Evaluate LibriPhrase
    print("\n" + "=" * 80)
    print("Evaluation Results")
    if args.spk_scale is not None or args.spk_bias is not None:
        cur_scale = args.spk_scale if args.spk_scale is not None else orig_scale
        cur_bias = args.spk_bias if args.spk_bias is not None else orig_bias
        print(f"  Calibration: scale={cur_scale}, bias={cur_bias}")
    print("=" * 80)

    loader_names = ['Easy', 'Hard']
    all_results = []

    for loader_idx, loader in enumerate(eval_loaders):
        name = loader_names[loader_idx] if loader_idx < len(loader_names) else f'Loader{loader_idx}'
        metrics, metrics_calc, pphon_stats, raw_data = evaluate_loader(
            model, loader, name, saved_args, device, collect_raw=collect_raw
        )
        all_results.append((name, metrics, metrics_calc, pphon_stats))
        print_metrics(name, metrics, pphon_stats)

        if collect_raw and raw_data is not None:
            # Back-compute cos_sim from P_spk using current (scale, bias)
            cur_scale = args.spk_scale if args.spk_scale is not None else orig_scale
            cur_bias = args.spk_bias if args.spk_bias is not None else orig_bias
            if cur_scale is not None and cur_bias is not None:
                raw_data['cos_sim'] = back_compute_cos_sim(raw_data['P_spk'], cur_scale, cur_bias)
            all_raw_data[name] = raw_data

        # Run diagnostics if requested
        if args.diagnostics:
            try:
                from score_diagnostics import dump_score_diagnostics
                dump_score_diagnostics(
                    metrics_calculator=metrics_calc,
                    dataset_name=name,
                    epoch=checkpoint.get('epoch', 0),
                    output_dir=output_dir,
                    save_raw=args.save_raw_scores,
                    fusion_mode=saved_args.get('fusion_mode', 'multiply'),
                )
            except Exception as e:
                print(f"  [WARNING] Score diagnostics failed: {e}")

    # Evaluate extended datasets
    if args.eval_gsc_qualcomm:
        print("\n>> Creating extended test loaders (GSC + Qualcomm)...")
        extended_loaders = create_extended_loaders(saved_args, args)

        for dataset_key, dataset_name in [
            ('google_speech_commands', 'GSC'),
            ('qualcomm', 'Qualcomm')
        ]:
            loader = extended_loaders.get(dataset_key)
            if loader is None:
                continue
            metrics, metrics_calc, pphon_stats, raw_data = evaluate_loader(
                model, loader, dataset_name, saved_args, device, collect_raw=collect_raw
            )
            all_results.append((dataset_name, metrics, metrics_calc, pphon_stats))
            print_metrics(dataset_name, metrics, pphon_stats)

            if collect_raw and raw_data is not None:
                cur_scale = args.spk_scale if args.spk_scale is not None else orig_scale
                cur_bias = args.spk_bias if args.spk_bias is not None else orig_bias
                if cur_scale is not None and cur_bias is not None:
                    raw_data['cos_sim'] = back_compute_cos_sim(raw_data['P_spk'], cur_scale, cur_bias)
                all_raw_data[dataset_name] = raw_data

            if args.diagnostics:
                try:
                    from score_diagnostics import dump_score_diagnostics
                    dump_score_diagnostics(
                        metrics_calculator=metrics_calc,
                        dataset_name=dataset_name,
                        epoch=checkpoint.get('epoch', 0),
                        output_dir=output_dir,
                        save_raw=args.save_raw_scores,
                        fusion_mode=saved_args.get('fusion_mode', 'multiply'),
                    )
                except Exception as e:
                    print(f"  [WARNING] Score diagnostics failed for {dataset_name}: {e}")

    # === Grid Search ===
    if args.grid_search and all_raw_data:
        scales = [float(x) for x in args.grid_scales.split(',')]
        biases = [float(x) for x in args.grid_biases.split(',')]
        fusion_mode = saved_args.get('fusion_mode', 'multiply')
        run_grid_search(all_raw_data, scales, biases, fusion_mode, output_dir)

    # Save evaluation report
    report_path = os.path.join(output_dir, "eval_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Evaluation Report\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Epoch: {checkpoint.get('epoch', 'N/A')}\n")
        f.write(f"Mode: {saved_args.get('mode')}\n")
        f.write(f"Audio Input: {saved_args.get('audio_input')}\n")
        f.write(f"Bidirectional: {saved_args.get('bidirectional', False)}\n")
        f.write(f"GRU Layers: {saved_args.get('gru_layers', 2)}\n")
        if args.spk_scale is not None or args.spk_bias is not None:
            f.write(f"Calibration Override: scale={args.spk_scale}, bias={args.spk_bias}\n")
        f.write(f"{'=' * 60}\n\n")

        for name, metrics, _, pphon_stats in all_results:
            f.write(f"[{name}]\n")
            for mode in ['C-KWS', 'TB-KWS', 'TO-KWS']:
                if mode in metrics and 'error' not in metrics[mode]:
                    m = metrics[mode]
                    f.write(
                        f"  {mode:8s} - EER: {m['EER']*100:6.2f}%  "
                        f"AUC: {m['AUC']*100:6.2f}%  "
                        f"FRR@1%: {m.get('FRR@FAR1%', 0)*100:6.2f}%  "
                        f"FRR@5%: {m.get('FRR@FAR5%', 0)*100:6.2f}%  "
                        f"FRR@10%: {m.get('FRR@FAR10%', 0)*100:6.2f}%\n"
                    )
            if 'SV' in metrics and 'error' not in metrics['SV']:
                m = metrics['SV']
                f.write(
                    f"  {'SV':8s} - EER: {m['EER']*100:6.2f}%  "
                    f"AUC: {m['AUC']*100:6.2f}%\n"
                )
            if pphon_stats:
                f.write(
                    f"  P_phon: pos_mean={pphon_stats['positive_mean']:.4f}, "
                    f"neg_mean={pphon_stats['negative_mean']:.4f}, "
                    f"gap={pphon_stats['gap_mean']:.4f}\n"
                )
            f.write("\n")

    print(f"\n>> Report saved to: {report_path}")
    if args.diagnostics:
        diag_dir = os.path.join(output_dir, 'diagnostics')
        if os.path.isdir(diag_dir):
            print(f">> Diagnostics saved to: {diag_dir}/")

    print("\n>> Evaluation complete!")


if __name__ == "__main__":
    main()