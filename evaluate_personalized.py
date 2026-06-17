"""
Evaluation Script for P-PhonMatchNet

Evaluates a trained P-PhonMatchNet model on personalized KWS tasks:
- C-KWS: Conventional keyword spotting
- TB-KWS: Target-biased keyword spotting
- TO-KWS: Target-only keyword spotting

Usage:
    python evaluate_personalized.py \\
        --checkpoint results_personalized/checkpoint_49.pth \\
        --mode TB-KWS \\
        --output_dir eval_results
"""

import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import model and dataset
from model import p_ukws
from dataset import personalized_libriphrase
from dataset import KWSDataLoader
from criterion.personalized_metrics import (
    PersonalizedKWSMetrics,
    format_metrics_report,
    save_metrics_to_file
)

import warnings
warnings.filterwarnings("ignore", message=".*weights_only.*")
warnings.filterwarnings("ignore", message=".*TorchScript.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate P-PhonMatchNet on personalized KWS tasks"
    )

    # Model and checkpoint
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='Path to model checkpoint')
    parser.add_argument('--mode', required=False, type=str, default='TB-KWS',
                        choices=['C-KWS', 'TB-KWS', 'TO-KWS'],
                        help='Evaluation mode')

    # Dataset configuration
    parser.add_argument('--batch_size', required=False, type=int, default=512,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', required=False, type=int, default=2,
                        help='Number of dataloader workers')
    
    # NOTE: The following parameters are automatically read from checkpoint:
    # --text_input, --audio_input, --stack_extractor
    # --frame_length, --hop_length, --sample_rate, --log_mel
    # --speaker_encoder_path, --disable_film, --disable_sv_branch

    # Personalized evaluation settings
    parser.add_argument('--personalized', action='store_true',
                        help='Enable personalized evaluation with enrollment audio')
    parser.add_argument('--speaker_ratio', required=False, type=float, default=0.5,
                        help='Ratio of same-speaker pairs')

    # Test dataset paths
    parser.add_argument('--test_pkl', required=False, type=str,
                        default='/padawan/test_500h.pkl',
                        help='Path to test pickle file')
    parser.add_argument('--test_types', required=False, type=str, default='both',
                        choices=['easy', 'hard', 'both'],
                        help='Test difficulty types')
    parser.add_argument('--google_pkl', required=False, type=str,
                        default='/padawan/google.pkl',
                        help='Path to Google Speech Commands pickle')
    parser.add_argument('--qualcomm_pkl', required=False, type=str,
                        default='/padawan/qualcomm.pkl',
                        help='Path to Qualcomm pickle')
    parser.add_argument('--eval_all', action='store_true',
                        help='Evaluate on all datasets (LibriPhrase + GSC + Qualcomm)')
    parser.add_argument('--datasets', nargs='+', default=['all'],
                        choices=['all', 'lpe', 'lph', 'google_speech_commands', 'qualcomm'],
                        help='Specific datasets to evaluate (overrides --eval_all implies): lpe, lph, google_speech_commands, qualcomm')

    # Output configuration
    parser.add_argument('--output_dir', required=False, type=str,
                        default='eval_results',
                        help='Output directory for results')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save ROC curve plots')
    parser.add_argument('--device', required=False, type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for evaluation')
    parser.add_argument('--diagnostic', action='store_true',
                        help='Enable diagnostic mode to print detailed analysis for model debugging')

    args = parser.parse_args()
    return args


def load_model(checkpoint_path: str, args, device: str = 'cuda'):
    """
    Load model from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        args: Evaluation arguments
        device: Device to load model on

    Returns:
        model: Loaded P_UKWS model
        vocab: Vocabulary size
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # ===== Read all model configuration from checkpoint =====
    if 'args' not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain training args!\n"
            "Please use a checkpoint saved with train_personalized.py"
        )
    
    train_args = checkpoint['args']
    
    # Extract vocab for backward compatibility
    vocab = train_args.get('vocab', 42)
    
    # Model configuration - ALL from checkpoint
    kwargs = {
        'vocab': vocab,
        'text_input': train_args['text_input'],
        'audio_input': train_args['audio_input'],
        'stack_extractor': train_args['stack_extractor'],
        'frame_length': train_args['frame_length'],
        'hop_length': train_args['hop_length'],
        'num_mel': 40,
        'sample_rate': train_args['sample_rate'],
        'log_mel': train_args['log_mel'],
        'mode': args.mode,  # ✅ Evaluation mode can be changed
        'speaker_encoder_path': train_args.get('speaker_encoder_path', 
                                               'model/speaker/efficient_tdnn'),
        'disable_film': train_args.get('disable_film', False),
        'disable_sv_branch': train_args.get('disable_sv_branch', False),
        'enrollment_dropout': 0.0,  # Always disable dropout during evaluation
    }
    
    # Initialize model
    print(f"Initializing P-PhonMatchNet in {args.mode} mode...")
    print(f"  Architecture from checkpoint:")
    print(f"    - audio_input: {kwargs['audio_input']}")
    print(f"    - text_input: {kwargs['text_input']}")
    print(f"    - stack_extractor: {kwargs['stack_extractor']}")
    print(f"    - sample_rate: {kwargs['sample_rate']}")
    if kwargs['disable_film']:
        print(f"    - [ABLATION] FiLM disabled")
    if kwargs['disable_sv_branch']:
        print(f"    - [ABLATION] SV branch disabled")
    
    model = p_ukws.P_UKWS(**kwargs)

    # Load weights
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    # Print model info
    param_counts = model.count_parameters()
    print(f"Model loaded successfully")
    print(f"  Total parameters: {param_counts['total']:,}")
    print(f"  Trainable: {param_counts['trainable']:,}")
    print(f"  Frozen: {param_counts['frozen']:,}")

    return model, vocab, train_args


def create_all_test_loaders(args, train_args):
    """
    Create test dataloaders for ALL evaluation datasets.

    Args:
        args: Evaluation arguments
        train_args: Training arguments from checkpoint

    Returns:
        dict: {
            'libriphrase_easy': DataLoader,
            'libriphrase_hard': DataLoader,
            'google_speech_commands': DataLoader,
            'qualcomm': DataLoader,
        }
    """
    from dataset import KWSDataLoader
    from dataset.personalized_libriphrase import PersonalizedLibriPhraseDataset
    from dataset.personalized_google import PersonalizedGoogleCommandsDataset
    from dataset.personalized_qualcomm import PersonalizedQualcommDataset

    # Google embedding directory
    if train_args['audio_input'] == "raw":
        gemb_dir = None
    else:
        gemb_dir = '/padawan/google_speech_embedding/DB'

    # Use 'spawn' to avoid LMDB segfault with multiprocessing
    import multiprocessing
    mp_context = multiprocessing.get_context('spawn') if args.num_workers > 0 else None

    loaders = {}
    
    # Determine which datasets to load
    target_datasets = set(args.datasets)
    if 'all' in target_datasets or args.eval_all:
        target_datasets = {'lpe', 'lph', 'google_speech_commands', 'qualcomm'}
    
    # === 1. LibriPhrase Easy ===
    if 'lpe' in target_datasets:
        print("\n>> Creating LibriPhrase Easy test loader...")
        lp_easy = PersonalizedLibriPhraseDataset(
            batch_size=args.batch_size,
            gemb_dir=gemb_dir,
            features=train_args['text_input'],
            train=False,
            types='easy',
            shuffle=False,
            pkl=args.test_pkl,
            frame_length=train_args['frame_length'],
            hop_length=train_args['hop_length'],
            personalized=args.personalized,
            speaker_ratio=args.speaker_ratio,
        )
        loaders['libriphrase_easy'] = KWSDataLoader(
            lp_easy, args.batch_size,
            shuffle=True, pin_memory=True, drop_last=True,
            num_workers=args.num_workers,
            multiprocessing_context=mp_context
        )

    # === 2. LibriPhrase Hard ===
    if 'lph' in target_datasets:
        print(">> Creating LibriPhrase Hard test loader...")
        lp_hard = PersonalizedLibriPhraseDataset(
            batch_size=args.batch_size,
            gemb_dir=gemb_dir,
            features=train_args['text_input'],
            train=False,
            types='hard',
            shuffle=False,
            pkl=args.test_pkl,
            frame_length=train_args['frame_length'],
            hop_length=train_args['hop_length'],
            personalized=args.personalized,
            speaker_ratio=args.speaker_ratio,
        )
        loaders['libriphrase_hard'] = KWSDataLoader(
            lp_hard, args.batch_size,
            shuffle=True, pin_memory=True, drop_last=True,
            num_workers=args.num_workers,
            multiprocessing_context=mp_context
        )

    # === 3. Google Speech Commands ===
    if 'google_speech_commands' in target_datasets:
        print(">> Creating Google Speech Commands test loader...")
        gsc = PersonalizedGoogleCommandsDataset(
            batch_size=args.batch_size,
            gemb_dir=gemb_dir,
            features=train_args['text_input'],
            testset_only=True,
            shuffle=False,
            pkl=args.google_pkl,
            frame_length=train_args['frame_length'],
            hop_length=train_args['hop_length'],
            personalized=args.personalized,
            speaker_ratio=args.speaker_ratio,
        )
        loaders['google_speech_commands'] = KWSDataLoader(
            gsc, args.batch_size,
            shuffle=True, pin_memory=True, drop_last=True,
            num_workers=args.num_workers,
            multiprocessing_context=mp_context
        )

    # === 4. Qualcomm ===
    if 'qualcomm' in target_datasets:
        print(">> Creating Qualcomm test loader...")
        qualcomm = PersonalizedQualcommDataset(
            batch_size=args.batch_size,
            gemb_dir=gemb_dir,
            features=train_args['text_input'],
            shuffle=False,
            pkl=args.qualcomm_pkl,
            frame_length=train_args['frame_length'],
            hop_length=train_args['hop_length'],
            personalized=args.personalized,
            speaker_ratio=args.speaker_ratio,
        )
        loaders['qualcomm'] = KWSDataLoader(
            qualcomm, args.batch_size,
            shuffle=True, pin_memory=True, drop_last=True,
            num_workers=args.num_workers,
            multiprocessing_context=mp_context
        )

    return loaders


def prepare_test_loader(args, train_args, vocab):
    """
    Prepare test dataloader

    Args:
        args: Evaluation arguments
        train_args: Training arguments from checkpoint
        vocab: Vocabulary size

    Returns:
        test_loader: Test dataloader
        test_dataset: Test dataset
    """
    # Google embedding directory
    if train_args['audio_input'] == "raw":
        gemb_dir = None
    else:
        gemb_dir = '/padawan/google_speech_embedding/DB'

    # Create test dataset with personalized pairing
    test_dataset = personalized_libriphrase.PersonalizedLibriPhraseDataset(
        batch_size=args.batch_size,
        gemb_dir=gemb_dir,
        features=train_args['text_input'],
        train=False,  # Test mode
        types=args.test_types,
        shuffle=False,  # Don't shuffle for evaluation
        pkl=args.test_pkl,
        frame_length=train_args['frame_length'],
        hop_length=train_args['hop_length'],
        personalized=args.personalized,  # Enable speaker pairing
        speaker_ratio=args.speaker_ratio,
    )

    # Use 'spawn' to avoid LMDB segfault with multiprocessing
    import multiprocessing
    mp_context = multiprocessing.get_context('spawn') if args.num_workers > 0 else None

    test_loader = KWSDataLoader(
        test_dataset,
        args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=args.num_workers,
        multiprocessing_context=mp_context
    )

    print(f"Test dataset prepared:")
    print(f"  Total samples: {len(test_dataset)}")
    print(f"  Personalized: {args.personalized}")

    if args.personalized:
        # Get category breakdown
        categories = []
        for i in range(min(1000, len(test_dataset))):  # Sample first 1000
            item = test_dataset[i]
            categories.append(item['category'])

        unique, counts = np.unique(categories, return_counts=True)
        print(f"  Category distribution (first 1000 samples):")
        for cat, count in zip(unique, counts):
            print(f"    {cat}: {count}")

    return test_loader, test_dataset


def evaluate_model(model, test_loader, args, train_args):
    """
    Evaluate model on test set

    Args:
        model: P_UKWS model
        test_loader: Test dataloader
        args: Evaluation arguments
        train_args: Training arguments from checkpoint

    Returns:
        metrics: Computed metrics
    """
    print(f"\nEvaluating model in {args.mode} mode...")

    # Initialize metrics
    metrics_calculator = PersonalizedKWSMetrics()

    # Evaluation loop
    device = args.device
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # Prepare inputs
            if train_args['audio_input'] == "raw":
                speech_input = batch["x"]
                speech_len = batch["x_len"]
            elif train_args['audio_input'] == "google_embed":
                speech_input = batch["gemb"]
                speech_len = batch["gemb_len"]
            elif train_args['audio_input'] == "both":
                speech_input = (batch["x"], batch["gemb"])
                speech_len = (batch["x_len"], batch["gemb_len"])

            # Forward pass
            # NOTE: args.mode does NOT affect forward pass
            if args.personalized:
                # Personalized: always need enrollment_audio
                # Because compute() evaluates ALL three modes
                output = model(
                    speech=speech_input,
                    text=batch["y"],
                    speech_len=speech_len,
                    text_len=batch["y_len"],
                    enrollment_audio=batch.get("enrollment_audio"),
                )
            else:
                # Standard PhonMatchNet: no enrollment
                output = model(
                    speech=speech_input,
                    text=batch["y"],
                    speech_len=speech_len,
                    text_len=batch["y_len"],
                )

            # Get predictions
            # Extract keyword and speaker scores separately
            # C-KWS needs P_utt only, TB-KWS/TO-KWS need P_utt × P_spk
            keyword_scores = output['P_utt']  # [B, 1] Always available

            if args.personalized:
                # Get speaker scores from model output
                speaker_scores = output.get('P_spk', torch.ones_like(keyword_scores))
            else:
                # Standard mode: no speaker verification (P_spk = 1.0)
                speaker_scores = torch.ones_like(keyword_scores)

            # Get labels
            # === CRITICAL FIX: Use z_keyword for C-KWS evaluation ===
            # In personalized datasets, batch['z'] is z_final (AND-gated with speaker)
            # We need z_keyword for proper C-KWS evaluation
            if args.personalized:
                keyword_labels = batch.get('z_keyword', batch['z'])
                speaker_labels = batch.get('speaker_label', torch.zeros_like(keyword_labels))
                categories = batch.get('category', ['ts-tk'] * len(keyword_labels))
            else:
                keyword_labels = batch['z']  # [B, 1] Keyword match
                speaker_labels = torch.zeros_like(keyword_labels)
                categories = ['ts-tk'] * len(keyword_labels)

            # Update metrics with separate scores
            metrics_calculator.update(
                keyword_scores=keyword_scores,
                speaker_scores=speaker_scores,
                keyword_labels=keyword_labels,
                speaker_labels=speaker_labels,
                categories=categories
            )

    # Compute final metrics
    print("\nComputing metrics...")
    metrics = metrics_calculator.compute()

    # Print report
    print("\n" + "=" * 80)
    print(format_metrics_report(metrics, dataset_name=f"Test ({args.test_types})"))

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Save metrics to file
    metrics_file = os.path.join(args.output_dir, f'metrics_{args.mode}.txt')
    save_metrics_to_file(metrics, metrics_file, dataset_name=f"Test ({args.test_types})")

    # Save ROC plots if requested
    if args.save_plots:
        plot_file = os.path.join(args.output_dir, f'roc_curves_{args.mode}.png')
        metrics_calculator.plot_roc_curves(save_path=plot_file)

    # Save category breakdown
    if args.personalized:
        breakdown = metrics_calculator.get_category_breakdown()
        print("\nCategory Breakdown:")
        for cat, count in sorted(breakdown.items()):
            print(f"  {cat}: {count}")

    print(f"\nResults saved to {args.output_dir}")

    return metrics


def evaluate_single_dataset(model, loader, dataset_name, args, train_args):
    """
    Evaluate model on a SINGLE dataset and print detailed report immediately.

    Args:
        model: P_UKWS model
        loader: Test dataloader
        dataset_name: Name of the dataset (e.g., 'LPE', 'LPH', 'GSC', 'Qualcomm')
        args: Evaluation arguments
        train_args: Training arguments from checkpoint

    Returns:
        metrics: Computed metrics dictionary (or None if failed)
    """
    from criterion.personalized_metrics import PersonalizedKWSMetrics

    try:
        # Initialize metrics for this dataset
        metrics_calculator = PersonalizedKWSMetrics()
        sample_count = 0

        # Evaluation loop
        model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating {dataset_name}"):
                # Move to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(args.device)

                # Prepare inputs
                if train_args['audio_input'] == "raw":
                    speech_input = batch["x"]
                    speech_len = batch["x_len"]
                elif train_args['audio_input'] == "google_embed":
                    speech_input = batch["gemb"]
                    speech_len = batch["gemb_len"]
                elif train_args['audio_input'] == "both":
                    speech_input = (batch["x"], batch["gemb"])
                    speech_len = (batch["x_len"], batch["gemb_len"])

                # Forward pass
                if args.personalized:
                    output = model(
                        speech=speech_input,
                        text=batch["y"],
                        speech_len=speech_len,
                        text_len=batch["y_len"],
                        enrollment_audio=batch.get("enrollment_audio"),
                    )
                else:
                    output = model(
                        speech=speech_input,
                        text=batch["y"],
                        speech_len=speech_len,
                        text_len=batch["y_len"],
                    )

                # Get predictions
                # Extract keyword and speaker scores separately
                keyword_scores = output['P_utt']  # [B, 1] Always available

                if args.personalized:
                    # Get speaker scores from model output
                    speaker_scores = output.get('P_spk', torch.ones_like(keyword_scores))
                else:
                    # Standard mode: no speaker verification (P_spk = 1.0)
                    speaker_scores = torch.ones_like(keyword_scores)

                # Get labels
                # === CRITICAL FIX: Use z_keyword for C-KWS evaluation ===
                # In personalized datasets, batch['z'] is z_final (AND-gated with speaker)
                # We need z_keyword for proper C-KWS evaluation
                if args.personalized:
                    # Use z_keyword for keyword labels (required for correct C-KWS)
                    keyword_labels = batch.get('z_keyword', batch['z'])
                    speaker_labels = batch.get('speaker_label', torch.zeros_like(keyword_labels))
                    categories = batch.get('category', ['ts-tk'] * len(keyword_labels))
                else:
                    keyword_labels = batch['z']
                    speaker_labels = torch.zeros_like(keyword_labels)
                    categories = ['ts-tk'] * len(keyword_labels)

                # Update metrics with separate scores
                metrics_calculator.update(
                    keyword_scores=keyword_scores,
                    speaker_scores=speaker_scores,
                    keyword_labels=keyword_labels,
                    speaker_labels=speaker_labels,
                    categories=categories
                )
                sample_count += len(keyword_labels)

        # Compute metrics
        metrics = metrics_calculator.compute()

        # === PRINT DETAILED REPORT IMMEDIATELY ===
        print("\n" + "="*80)
        print(f"Dataset: {dataset_name}")
        print("="*80)
        print(f"Samples: {sample_count}")
        print()

        for mode in ['C-KWS', 'TB-KWS', 'TO-KWS']:
            mode_full_name = {
                'C-KWS': 'Conventional KWS',
                'TB-KWS': 'Target-Biased KWS',
                'TO-KWS': 'Target-Only KWS'
            }[mode]

            if mode in metrics and 'EER' in metrics[mode]:
                m = metrics[mode]
                eer = m['EER'] * 100
                auc = m.get('AUC', 0) * 100
                frr1 = m.get('FRR@FAR1%', 0) * 100
                frr5 = m.get('FRR@FAR5%', 0) * 100
                frr10 = m.get('FRR@FAR10%', 0) * 100

                print(f"[{mode}] {mode_full_name}")
                print(f"  EER: {eer:5.2f}% | AUC: {auc:5.2f}%")
                print(f"  FRR @ FAR 1%: {frr1:5.2f}% | @ 5%: {frr5:5.2f}% | @ 10%: {frr10:5.2f}%")
                print()

        # === DIAGNOSTIC MODE: Print detailed analysis ===
        if getattr(args, 'diagnostic', False):
            diagnostic_report = metrics_calculator.get_diagnostic_report(mode='TO-KWS')
            print(diagnostic_report)
            
            # Save diagnostic report to file
            diag_file = os.path.join(args.output_dir, f'diagnostic_{dataset_name}.txt')
            os.makedirs(args.output_dir, exist_ok=True)
            with open(diag_file, 'w') as f:
                f.write(diagnostic_report)
            print(f"\nDiagnostic report saved to {diag_file}")

        return metrics

    except Exception as e:
        print(f"\n❌ ERROR evaluating {dataset_name}:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_all_datasets(model, test_loaders, args, train_args):
    """
    Evaluate model on ALL test datasets with independent evaluation and immediate feedback.

    Args:
        model: P_UKWS model
        test_loaders: Dictionary of test dataloaders
        args: Evaluation arguments
        train_args: Training arguments from checkpoint

    Returns:
        results: {
            'LPE': metrics_dict or None,
            'LPH': metrics_dict or None,
            'GSC': metrics_dict or None,
            'Qualcomm': metrics_dict or None,
        }
    """
    # Dataset name mapping
    dataset_names = {
        'libriphrase_easy': 'LibriPhrase-Easy (LPE)',
        'libriphrase_hard': 'LibriPhrase-Hard (LPH)',
        'google_speech_commands': 'Google Speech Commands (GSC)',
        'qualcomm': 'Qualcomm'
    }

    all_results = {}

    # Evaluate each dataset independently
    for key, loader in test_loaders.items():
        dataset_name = dataset_names.get(key, key)
        result = evaluate_single_dataset(model, loader, dataset_name, args, train_args)

        # Store result (even if None for failed datasets)
        short_name = key.replace('libriphrase_easy', 'LPE').replace('libriphrase_hard', 'LPH').replace('google_speech_commands', 'GSC').replace('qualcomm', 'Qualcomm')
        all_results[short_name] = result

    return all_results


def print_summary_table(all_results):
    """
    Print summary table with EER and FRR@1% for all datasets and modes.

    Args:
        all_results: Dictionary of results from evaluate_all_datasets
    """
    print("\n" + "="*95)
    print("Summary Table: Personalized KWS Performance")
    print("="*95)
    print(f"{'Dataset':<12} | {'C-KWS':<6} | {'TB-KWS':<15} | {'TO-KWS':<15} |")
    print(f"{'':12} | {'EER':<6} | {'EER':<6} {'FRR@1%':<8} | {'EER':<6} {'FRR@1%':<8} |")
    print("-"*95)

    # Dataset order
    dataset_order = ['LPE', 'LPH', 'GSC', 'Qualcomm']

    for dataset_key in dataset_order:
        metrics = all_results.get(dataset_key)

        if metrics is None:
            # Dataset failed
            print(f"{dataset_key:<12} | {'N/A':<6} | {'N/A':<6} {'N/A':<8} | {'N/A':<6} {'N/A':<8} |")
            continue

        # Extract metrics
        c_eer = metrics.get('C-KWS', {}).get('EER', 0) * 100 if 'C-KWS' in metrics else 0
        tb_eer = metrics.get('TB-KWS', {}).get('EER', 0) * 100 if 'TB-KWS' in metrics else 0
        tb_frr1 = metrics.get('TB-KWS', {}).get('FRR@FAR1%', 0) * 100 if 'TB-KWS' in metrics else 0
        to_eer = metrics.get('TO-KWS', {}).get('EER', 0) * 100 if 'TO-KWS' in metrics else 0
        to_frr1 = metrics.get('TO-KWS', {}).get('FRR@FAR1%', 0) * 100 if 'TO-KWS' in metrics else 0

        print(f"{dataset_key:<12} | {c_eer:5.2f}  | {tb_eer:5.2f}  {tb_frr1:7.2f}  | {to_eer:5.2f}  {to_frr1:7.2f}  |")

    print("="*95)


def save_all_results(results, output_dir):
    """
    Save comprehensive results for all datasets.

    Args:
        results: Dictionary of results from evaluate_all_datasets
        output_dir: Output directory
    """
    import json
    from criterion.personalized_metrics import save_metrics_to_file

    os.makedirs(output_dir, exist_ok=True)

    # Save each dataset's results
    for dataset_name, metrics in results.items():
        if metrics is None:
            # Skip failed datasets
            print(f"⚠️ Skipping save for {dataset_name} (evaluation failed)")
            continue

        metrics_file = os.path.join(output_dir, f'metrics_{dataset_name}.txt')
        save_metrics_to_file(metrics, metrics_file, dataset_name=dataset_name)

    # Save summary JSON
    summary = {}
    for dataset_name, metrics in results.items():
        if metrics is None:
            summary[dataset_name] = "Failed"
            continue

        summary[dataset_name] = {}
        for mode in ['C-KWS', 'TB-KWS', 'TO-KWS']:
            if mode in metrics:
                summary[dataset_name][mode] = {
                    'EER': float(metrics[mode].get('EER', 0)),
                    'AUC': float(metrics[mode].get('AUC', 0)),
                    'FRR@FAR1%': float(metrics[mode].get('FRR@FAR1%', 0)),
                    'FRR@FAR5%': float(metrics[mode].get('FRR@FAR5%', 0)),
                    'FRR@FAR10%': float(metrics[mode].get('FRR@FAR10%', 0)),
                }

    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll results saved to {output_dir}")
    print(f"  - Individual metrics: metrics_{{dataset}}.txt")
    print(f"  - Summary JSON: summary.json")


def main():
    """Main evaluation function"""
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Load model and training config
    model, vocab, train_args = load_model(args.checkpoint, args, device=args.device)

    if args.eval_all or args.datasets != ['all']:
        # Evaluate on ALL datasets
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION - All Datasets")
        print("="*80)

        test_loaders = create_all_test_loaders(args, train_args)
        results = evaluate_all_datasets(model, test_loaders, args, train_args)

        # Print summary table
        print_summary_table(results)

        # Save comprehensive results
        save_all_results(results, args.output_dir)

        print("\n✅ Evaluation complete on all datasets!")
        return results
    else:
        # Original single-dataset evaluation
        test_loader, test_dataset = prepare_test_loader(args, train_args, vocab)
        metrics = evaluate_model(model, test_loader, args, train_args)

        print("\n✅ Evaluation complete!")
        return metrics


if __name__ == "__main__":
    main()
