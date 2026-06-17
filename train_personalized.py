"""
Training Script for P-PhonMatchNet (Personalized Keyword Spotting)

Extends the baseline training script to support personalized KWS with
speaker verification.

Key Features:
- Supports both personalized and standard training modes
- Frozen speaker encoder (899K params)
- Same loss as baseline: L_total = L_utt + L_phon
- Three modes: C-KWS, TB-KWS, TO-KWS

Usage:
    # Standard training (C-KWS mode)
    python train_personalized.py --mode C-KWS --epoch 50 --lr 1e-4

    # Personalized training (TB-KWS mode)
    python train_personalized.py --mode TB-KWS --epoch 50 --lr 1e-4 --personalized
"""

import argparse
import logging
import os
from pathlib import Path
from tqdm import tqdm

# Download required NLTK resources automatically
import nltk
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    print(">> Downloading required NLTK resources...")
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('punkt', quiet=True)
    print(">> NLTK resources downloaded successfully!")


import numpy as np
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, LoggerType
from datetime import datetime

# Import P-PhonMatchNet model
from model import p_ukws
from dataset import personalized_libriphrase
from dataset import KWSDataLoader
from dataset.personalized_google import PersonalizedGoogleCommandsDataset
from dataset.personalized_qualcomm import PersonalizedQualcommDataset
from criterion import total
from criterion.utils import eer, compute_eer
from criterion.personalized_metrics import PersonalizedKWSMetrics

# AGC (Alignment-Guided Contrastive) for frame-level mismatch weighting
from utils.phoneme_mapping import build_g2p_to_mfa_mapping, get_g2p_phonemes, load_mfa_vocab, MFA_VOCAB
from utils.agc import compute_frame_mismatch_mask

from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import BinaryAUROC
import gc
import warnings
import logging as base_logging

# ===== 抑制各種 Warning =====
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")
warnings.filterwarnings("ignore", message=".*weights_only.*")
warnings.filterwarnings("ignore", message=".*TorchScript.*")
warnings.filterwarnings("ignore", message=".*does not have many workers.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 抑制 transformers / tqdm 等的 logging
base_logging.getLogger("transformers").setLevel(base_logging.ERROR)
base_logging.getLogger("tqdm").setLevel(base_logging.ERROR)

# 設定環境變數抑制 TF 和 ONNX 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False

seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)
logger = get_logger(__name__, log_level="INFO")


# ===== P_phon 輕量統計函數 =====
def compute_pphon_lightweight_stats(pos_means, pos_mins, neg_means, neg_mins):
    """計算 P_phon 輕量統計"""
    pos_mean_all = torch.cat(pos_means).mean().item() if pos_means else 0
    pos_min_all = torch.cat(pos_mins).mean().item() if pos_mins else 0
    neg_mean_all = torch.cat(neg_means).mean().item() if neg_means else 0
    neg_min_all = torch.cat(neg_mins).mean().item() if neg_mins else 0
    
    return {
        'positive_mean': pos_mean_all,
        'positive_min': pos_min_all,
        'negative_mean': neg_mean_all,
        'negative_min': neg_min_all,
        'gap_mean': pos_mean_all - neg_mean_all,
        'gap_min': pos_min_all - neg_min_all,
    }


def print_pphon_lightweight_summary(stats, loader_name=""):
    """印出 P_phon 輕量摘要（一行）"""
    prefix = f"[{loader_name}] " if loader_name else ""
    print(f"  {prefix}P_phon: pos_mean={stats['positive_mean']:.3f}, "
          f"neg_mean={stats['negative_mean']:.3f}, "
          f"gap={stats['gap_mean']:.3f} | "
          f"pos_min={stats['positive_min']:.3f}, "
          f"neg_min={stats['negative_min']:.3f}, "
          f"gap={stats['gap_min']:.3f}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Training script for P-PhonMatchNet"
    )

    # Training hyperparameters
    parser.add_argument('--epoch', required=True, type=int,
                        help='Number of training epochs')
    parser.add_argument('--lr', required=True, type=float,
                        help='Learning rate')
    parser.add_argument('--batch_size', required=False, type=int, default=64,
                        help='Batch size')
    parser.add_argument('--num_workers', required=False, type=int, default=2,
                        help='Number of dataloader workers')
    parser.add_argument('--loss_weight', default=[1.0, 1.0], nargs=2, type=float,
                        help='Loss weights for [L_utt, L_phon]')

    # Model configuration
    parser.add_argument('--mode', required=False, type=str, default='TO-KWS',
                        choices=['C-KWS', 'TB-KWS', 'TO-KWS'],
                        help='Primary mode for early stopping (all modes are always evaluated during validation)')
    parser.add_argument('--text_input', required=False, type=str, default='g2p_embed',
                        help='Text input type')
    parser.add_argument('--audio_input', required=False, type=str, default='raw',
                        choices=['raw', 'google_embed', 'both', 'enhanced_gembed'],
                        help='Audio input type')
    parser.add_argument('--stack_extractor', action='store_true',
                        help='Use stacked self-attention extractor')
    parser.add_argument('--audio_noise', action='store_true', default=True,
                        help='Apply noise augmentation')
    parser.add_argument('--disable_hybrid_encoder', action='store_true',
                        help='[Ablation] Disable HybridAudioEncoder (GRU) and use old EfficientAudioEncoder')
    parser.add_argument('--disable_ldn_norm', action='store_true',
                        help='[Ablation] Disable LayerNorm on LDN features (check energy alignment impact)')
    parser.add_argument('--gru_layers', type=int, default=2,
                        help='Number of GRU layers in AudioEncoder (1 or 2, default: 2)')
    parser.add_argument('--bidirectional', action='store_true',
                        help='Use bidirectional GRU in AudioEncoder')

    # Audio processing
    parser.add_argument('--frame_length', required=False, type=int, default=400,
                        help='Frame length for mel spectrogram')
    parser.add_argument('--hop_length', required=False, type=int, default=160,
                        help='Hop length for mel spectrogram')
    parser.add_argument('--sample_rate', required=False, type=int, default=16000,
                        help='Audio sample rate')
    parser.add_argument('--log_mel', action='store_true',
                        help='Use log mel spectrogram')

    # Personalized KWS settings
    parser.add_argument('--personalized', action='store_true',
                        help='Enable personalized training with enrollment audio')
    parser.add_argument('--speaker_ratio', required=False, type=float, default=0.5,
                        help='Ratio of same-speaker pairs (default: 0.5)')
    parser.add_argument('--speaker_encoder_path', required=False, type=str,
                        default='model/speaker/efficient_tdnn',
                        help='Path to speaker encoder weights')
    parser.add_argument('--personal_loss_weight', required=False, type=float, default=1.0,
                        help='Weight for L_personal (default: 1.0)')

    # Dataset paths
    parser.add_argument('--train_pkl', required=False, type=str,
                        default='/padawan/train_combined.pkl',
                        help='Path to training pickle file')
    parser.add_argument('--google_pkl', required=False, type=str,
                        default='/padawan/google_speech_commands/google.pkl',
                        help='Path to Google test pickle')
    parser.add_argument('--qualcomm_pkl', required=False, type=str,
                        default='/padawan/qualcomm/qualcomm.pkl',
                        help='Path to Qualcomm test pickle')
    parser.add_argument('--libriphrase_pkl', required=False, type=str,
                        default='/padawan/test_500h.pkl',
                        help='Path to LibriPhrase test pickle')

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_personalized",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="p_phonmatchnet",
        help="TensorBoard log directory",
    )
    parser.add_argument('--comment', required=False, type=str,
                        help='Experiment comment')
    parser.add_argument('--max_train_samples', type=int, default=None,
                    help='Limit training samples for mini test')
    parser.add_argument('--max_val_samples', type=int, default=None,
                    help='Limit validation samples for mini test')

    # ===== Ablation Study 開關 =====
    parser.add_argument('--disable_film', action='store_true',
                        help='[Ablation] Disable FiLM modulation (fix γ=1, β=0)')
    parser.add_argument('--disable_sv_branch', action='store_true',
                        help='[Ablation] Disable Speaker Verification branch (P_spk=1)')
    parser.add_argument('--unfreeze_speaker_encoder', action='store_true',
                        help='[Ablation] Unfreeze speaker encoder for fine-tuning')
    parser.add_argument('--finetuned_speaker_encoder', type=str, default=None,
                        help='Path to finetuned speaker encoder checkpoint (.pt file)')
    parser.add_argument('--film_target', type=str, default='fused',
                        choices=['fused', 'conv_only'],
                        help='FiLM modulation target: fused (default) or conv_only')
    parser.add_argument('--film_gate_type', type=str, default='pspk',
                        choices=['pspk', 'learned_scalar', 'learned_channel'],
                        help='FiLM gate type: pspk (v5.1), learned_scalar (v5.2), learned_channel (v5.2)')
    parser.add_argument('--film_lr_mult', type=float, default=0.1,
                        help='FiLM learning rate multiplier (e.g., 0.1 = 10%% of base LR, 1.0 = same as base LR)')
    parser.add_argument('--calib_lr_mult', type=float, default=1.0,
                        help='Calibration (spk_scale/spk_bias) LR multiplier (e.g., 10.0 = 10x base LR)')

    # === 實驗配置參數 ===
    parser.add_argument('--optimizer_type', required=False, type=str, default='adam',
                        choices=['adam', 'adam_wd', 'adamw'],
                        help="Optimizer type: 'adam' (baseline, no weight_decay), 'adam_wd' (Adam + weight_decay), 'adamw' (AdamW)")
    parser.add_argument('--scheduler_type', required=False, type=str, default='none',
                        choices=['none', 'plateau', 'cosine'],
                        help="Scheduler type: 'none' (baseline), 'plateau' (ReduceLROnPlateau), 'cosine' (CosineAnnealingLR)")
    parser.add_argument('--weight_decay', required=False, type=float, default=1e-4,
                        help="Weight decay value for adam_wd and adamw")

    # === Extended Evaluation Settings ===
    parser.add_argument('--eval_gsc_qualcomm', action='store_true', default=True,
                        help='Evaluate on GSC and Qualcomm datasets after LibriPhrase validation (default: True)')
    parser.add_argument('--no_eval_gsc_qualcomm', dest='eval_gsc_qualcomm', action='store_false',
                        help='Skip GSC and Qualcomm evaluation to save time')

    # === Focal Loss 參數 (B2/B3 Ablation Study) ===
    parser.add_argument('--focal_loss_phon', action='store_true', default=False,
                        help='[B2] Use Focal Loss for L_phon (L_utt remains BCE)')
    parser.add_argument('--focal_loss_utt', action='store_true', default=False,
                        help='[B3] Use Focal Loss for L_utt (L_phon remains BCE)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss gamma parameter (focusing strength)')
    parser.add_argument('--focal_alpha', type=float, default=None,
                        help='Focal Loss alpha (class balance weight), None=no balancing')

    # === Weighted PCL 參數 (B4 Ablation Study) ===
    parser.add_argument('--use_weighted_pcl', action='store_true', default=False,
                        help='[B4] Use Hard-Label Weighted PCL loss')
    parser.add_argument('--pcl_hard_weight', type=float, default=2.0,
                        help='Weight multiplier for hard negatives in PCL')
    parser.add_argument('--pcl_weight', type=float, default=1.0,
                        help='Weight for PCL loss in total loss')
    
    # === PCL 改進參數 (P1/P2) ===
    parser.add_argument('--pcl_version', type=str, default='v4',
                        choices=['v4', 'dynamic', 'margin'],
                        help='PCL version: v4=bug-fixed static, dynamic=P1 online hard mining, margin=P2 margin-based')
    parser.add_argument('--pcl_hard_scale', type=float, default=3.0,
                        help='[P1/P2] Dynamic hardness scale factor')
    parser.add_argument('--pcl_margin_pos', type=float, default=0.8,
                        help='[P2] Positive margin (s should be above this)')
    parser.add_argument('--pcl_margin_neg', type=float, default=0.2,
                        help='[P2] Negative margin (s should be below this)')
    
    # === P3: Phoneme-Position PCL 參數 ===
    parser.add_argument('--use_phoneme_pcl', action='store_true', default=False,
                        help='[P3] Add phoneme-position contrastive loss')
    parser.add_argument('--phoneme_pcl_weight', type=float, default=1,
                        help='[P3] Weight for phoneme-position PCL')
    parser.add_argument('--phoneme_pcl_gamma', type=float, default=2.0,
                        help='[P3] Focusing parameter (higher = more focus on hard phonemes)')
    parser.add_argument('--phoneme_pcl_margin', type=float, default=0.2,
                        help='[P3] Negative mismatch margin (P_phon should be below this)')

    # === MFA Auxiliary CE Loss 參數 ===
    parser.add_argument('--enable_aux_ce', action='store_true', default=False,
                        help='Enable MFA-based Auxiliary CE Loss for frame-level phoneme supervision')
    parser.add_argument('--lambda_aux', type=float, default=1,
                        help='Weight for Auxiliary CE Loss (search: 0.1, 0.3, 0.5, 1.0)')
    parser.add_argument('--frame_labels_path', type=str, default=None,
                        help='Path to frame_labels.lmdb generated by textgrid_to_framelabels.py')
    parser.add_argument('--n_phonemes', type=int, default=42,
                        help='Number of phoneme classes for Aux CE (42 = PAD + SIL + SPN + 39 ARPAbet)')
    parser.add_argument('--alpha_agc', type=float, default=1.0,
                        help='Weight for Alignment-guided Contrastive (L_agc) mismatch frames')
    parser.add_argument('--aux_label_smoothing', type=float, default=0.1,
                        help='Label smoothing for Aux CE (handles MFA boundary noise)')
    
    # === A5: Gembed Feature Dropout ===
    parser.add_argument('--gemb_drop_rate', type=float, default=0.0,
                        help='A5: Gembed feature dropout rate (0=off, 0.5=recommended)')
    
    # === A6: Curriculum Gembed ===
    parser.add_argument('--gemb_curriculum', action='store_true', default=False,
                        help='A6: Enable curriculum gembed introduction')
    parser.add_argument('--gemb_warmup_epochs', type=int, default=5,
                        help='A6: Phase 1 LDN-only epochs before introducing gembed')
    parser.add_argument('--gemb_ramp_epochs', type=int, default=10,
                        help='A6: Phase 2 epochs to linearly ramp gembed from 0 to 1')

    # === A2: Stream Fusion Mode ===
    parser.add_argument('--stream_fusion', type=str, default='add',
                        choices=['add', 'concat', 'gated'],
                        help='Audio stream fusion: add, concat, or gated (A6: gembed-guided gate)')

    # === AsymmetricMinPCL 參數 ===
    parser.add_argument('--use_neg_min_pcl', action='store_true', default=False,
                        help='Enable AsymmetricMinPCL (negative-only with positive floor)')
    parser.add_argument('--neg_min_pcl_weight', type=float, default=0.3,
                        help='Weight for AsymmetricMinPCL (v2 default: 0.3)')
    parser.add_argument('--neg_min_pcl_margin', type=float, default=0.2,
                        help='Negative margin: min(P_phon) should be below this')
    parser.add_argument('--neg_min_pcl_margin_pos', type=float, default=0.55,
                        help='Positive safety floor: min(P_phon) should stay above this')
    parser.add_argument('--neg_min_pcl_hard_scale', type=float, default=3.0,
                        help='Dynamic hardness scale for negatives')
    parser.add_argument('--neg_min_pcl_no_pos_floor', action='store_true', default=False,
                        help='Disable positive safety floor (for ablation)')

    # === Calibration Layer Ablation ===
    parser.add_argument('--calibration_mode', type=str, default='full',
                        choices=['linear', 'raw_sigmoid', 'frozen_init',
                                 'scale_only', 'bias_only', 'full'],
                        help='Calibration layer ablation mode')

    # === Score Fusion Mode ===
    parser.add_argument('--fusion_mode', type=str, default='multiply',
                        choices=['multiply', 'harmonic', 'min'],
                        help='Score fusion: multiply (default), harmonic, or min')

    # === Score-level supervision (anti-drift) ===
    parser.add_argument('--use_score_loss', action='store_true', default=False,
                        help='Enable L_score (content-aware anti-drift for calibration)')
    parser.add_argument('--score_loss_weight', type=float, default=1,
                        help='Weight for L_score (supplementary to L_personal)')

    args = parser.parse_args()
    return args


def prepare_loader(args):
    """
    Prepare dataloaders for training and validation

    Args:
        args: Command line arguments

    Returns:
        train_loader: Training dataloader
        eval_loaders: List of validation dataloaders
        vocab: Vocabulary size
        train_len: Training dataset length
    """
    # Use personalized dataset creation function
    train_loader, eval_loaders, vocab, train_len = \
        personalized_libriphrase.create_personalized_dataloaders(
            args,
            train_personalized=args.personalized
        )

    return train_loader, eval_loaders, vocab, train_len


def create_extended_test_loaders(args, train_args_dict):
    """
    Create test dataloaders for GSC and Qualcomm datasets.
    
    Args:
        args: Command line arguments
        train_args_dict: Training arguments dictionary (from checkpoint or current args)
    
    Returns:
        dict: {
            'google_speech_commands': DataLoader or None,
            'qualcomm': DataLoader or None,
        }
    """
    # Google embedding directory
    if train_args_dict.get('audio_input', 'raw') == "raw":
        gemb_dir = None
    else:
        gemb_dir = '/padawan/google_speech_embedding/DB'
    
    loaders = {}
    
    # === GSC ===
    try:
        logger.info(">> Creating Google Speech Commands test loader...")
        gsc = PersonalizedGoogleCommandsDataset(
            batch_size=args.batch_size,
            gemb_dir=gemb_dir,
            features=train_args_dict.get('text_input', 'both'),
            testset_only=False,  # Match baseline PhonMatchNet (full dataset)
            shuffle=False,
            pkl=args.google_pkl,
            frame_length=train_args_dict.get('frame_length', 400),
            hop_length=train_args_dict.get('hop_length', 160),
            personalized=args.personalized,
            speaker_ratio=args.speaker_ratio,
        )
        # NOTE: Using num_workers directly without spawn context (same as baseline train.py)
        loaders['google_speech_commands'] = KWSDataLoader(
            gsc, args.batch_size,
            shuffle=True, pin_memory=True, drop_last=True,
            num_workers=args.num_workers,
        )
        logger.info(f"   GSC loader created: {len(gsc)} samples")
    except Exception as e:
        logger.warning(f"Failed to create GSC loader: {e}")
        loaders['google_speech_commands'] = None
    
    # === Qualcomm ===
    try:
        logger.info(">> Creating Qualcomm test loader...")
        qualcomm = PersonalizedQualcommDataset(
            batch_size=args.batch_size,
            gemb_dir=gemb_dir,
            features=train_args_dict.get('text_input', 'both'),
            shuffle=False,
            pkl=args.qualcomm_pkl,
            frame_length=train_args_dict.get('frame_length', 400),
            hop_length=train_args_dict.get('hop_length', 160),
            personalized=args.personalized,
            speaker_ratio=args.speaker_ratio,
        )
        # NOTE: Using num_workers directly without spawn context (same as baseline train.py)
        loaders['qualcomm'] = KWSDataLoader(
            qualcomm, args.batch_size,
            shuffle=True, pin_memory=True, drop_last=True,
            num_workers=args.num_workers,
        )
        logger.info(f"   Qualcomm loader created: {len(qualcomm)} samples")
    except Exception as e:
        logger.warning(f"Failed to create Qualcomm loader: {e}")
        loaders['qualcomm'] = None
    
    return loaders


def evaluate_extended_dataset(model, loader, dataset_name, args, accelerator):
    """
    Evaluate model on an extended dataset (GSC or Qualcomm).
    
    Args:
        model: P_UKWS model (already prepared by accelerator)
        loader: Test dataloader
        dataset_name: Name of dataset ('GSC' or 'Qualcomm')
        args: Command line arguments
        accelerator: Accelerator instance
    
    Returns:
        metrics: Dictionary with C-KWS, TB-KWS, TO-KWS metrics, or None if failed
    """
    if loader is None:
        return None
    
    try:
        metrics_calculator = PersonalizedKWSMetrics(fusion_mode=getattr(args, 'fusion_mode', 'multiply'))
        model.eval()
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Eval {dataset_name}", 
                             disable=not accelerator.is_local_main_process,
                             dynamic_ncols=True, leave=False):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(accelerator.device)
                
                # Prepare inputs
                if args.audio_input == "raw":
                    speech_input = batch["x"]
                    speech_len = batch["x_len"]
                elif args.audio_input == "google_embed":
                    speech_input = batch["gemb"]
                    speech_len = batch["gemb_len"]
                elif args.audio_input in ("both", "enhanced_gembed"):
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
                        raw_audio_for_spk=batch.get("x")
                    )
                else:
                    output = model(
                        speech=speech_input,
                        text=batch["y"],
                        speech_len=speech_len,
                        text_len=batch["y_len"],
                    )
                
                # Extract scores
                keyword_scores = output['P_utt']
                if args.personalized:
                    speaker_scores = output.get('P_spk', torch.ones_like(keyword_scores))
                else:
                    speaker_scores = torch.ones_like(keyword_scores)
                
                # Get labels
                if args.personalized:
                    keyword_labels = batch.get('z_keyword', batch['z'])
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
        
        return metrics_calculator.compute()
    
    except Exception as e:
        logger.warning(f"Failed to evaluate {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir
    )

    accelerator = Accelerator(
        log_with=LoggerType.TENSORBOARD,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    set_seed(seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Prepare dataloaders
    train_dataloader, eval_dataloader, vocab, train_len = prepare_loader(args)

    # Model configuration
    kwargs = {
        'vocab': vocab,
        'text_input': args.text_input,
        'audio_input': args.audio_input,
        'stack_extractor': args.stack_extractor,
        'frame_length': args.frame_length,
        'hop_length': args.hop_length,
        'num_mel': 40,
        'sample_rate': args.sample_rate,
        'log_mel': args.log_mel,
        'mode': args.mode,
        'speaker_encoder_path': args.speaker_encoder_path,
        # ===== Ablation 開關 =====
        'disable_film': getattr(args, 'disable_film', False),
        'disable_sv_branch': getattr(args, 'disable_sv_branch', False),
        'freeze_speaker_encoder': not getattr(args, 'unfreeze_speaker_encoder', False),
        'finetuned_speaker_encoder_path': getattr(args, 'finetuned_speaker_encoder', None),
        'film_target': getattr(args, 'film_target', 'fused'),
        'film_gate_type': getattr(args, 'film_gate_type', 'pspk'),
        # MFA Auxiliary CE
        'enable_aux_ce': getattr(args, 'enable_aux_ce', False),
        'n_phonemes': getattr(args, 'n_phonemes', 42),
        # A5: Gembed Feature Dropout
        'gemb_drop_rate': getattr(args, 'gemb_drop_rate', 0.0),
        # A6: Curriculum Gembed
        'gemb_curriculum': getattr(args, 'gemb_curriculum', False),
        'gemb_warmup_epochs': getattr(args, 'gemb_warmup_epochs', 5),
        'gemb_ramp_epochs': getattr(args, 'gemb_ramp_epochs', 10),
        # Calibration Layer Ablation
        'calibration_mode': getattr(args, 'calibration_mode', 'full'),
        # Score Fusion Mode
        'fusion_mode': getattr(args, 'fusion_mode', 'multiply'),
        # Audio Encoder Ablation
        'disable_hybrid_encoder': getattr(args, 'disable_hybrid_encoder', False),
        'disable_ldn_norm': getattr(args, 'disable_ldn_norm', False),
        'gru_layers': getattr(args, 'gru_layers', 2),
        'bidirectional': getattr(args, 'bidirectional', False),
        # A2: Stream Fusion Mode
        'stream_fusion': getattr(args, 'stream_fusion', 'add'),
    }

    # Initialize P-PhonMatchNet model
    logger.info(f"Initializing P-PhonMatchNet in {args.mode} mode...")
    model = p_ukws.P_UKWS(**kwargs)
    model.to(accelerator.device)

    # Log model info
    param_counts = model.count_parameters()
    logger.info(f"Model Parameters:")
    logger.info(f"  Total: {param_counts['total']:,}")
    logger.info(f"  Trainable: {param_counts['trainable']:,}")
    logger.info(f"  Frozen (Speaker Encoder): {param_counts['frozen']:,}")

    # Loss functions (same as baseline)
    loss_object = total.TotalLoss(weight=args.loss_weight[0])
    loss_object_sce = total.TotalLoss_SCE(weight=args.loss_weight)

    # === NEW: BCELoss for loss_personal (直接監督 score) ===
    # 使用 reduction='none' 以便做 sample-level masking
    criterion_bce = nn.BCELoss(reduction='none')

    # === Focal Loss 初始化 (B2/B3 Ablation Study) ===
    criterion_phon_focal = None
    criterion_utt_focal = None
    
    if args.focal_loss_phon:
        from model.losses import PhonemeAwareFocalLoss
        criterion_phon_focal = PhonemeAwareFocalLoss(
            gamma=args.focal_gamma,
            alpha=args.focal_alpha
        )
        logger.info(f"📌 [B2] Using Focal Loss for L_phon (gamma={args.focal_gamma}, alpha={args.focal_alpha})")
    
    if args.focal_loss_utt:
        from model.losses import FocalLoss
        criterion_utt_focal = FocalLoss(
            gamma=args.focal_gamma,
            alpha=args.focal_alpha,
            reduction='sum'  # 改為 'sum' 以配合外層的 / batch_size
        )
        logger.info(f"📌 [B3] Using Focal Loss for L_utt (gamma={args.focal_gamma}, alpha={args.focal_alpha})")

    # === Weighted PCL 初始化 (B4 Ablation Study + P1/P2 改進) ===
    criterion_pcl = None
    if args.use_weighted_pcl:
        if args.pcl_version == 'dynamic':
            from model.losses import DynamicHardnessPCL
            criterion_pcl = DynamicHardnessPCL(
                hard_scale=args.pcl_hard_scale
            )
            logger.info(f"📌 [P1] Using Dynamic Hardness PCL (scale={args.pcl_hard_scale}, weight={args.pcl_weight})")
        elif args.pcl_version == 'margin':
            from model.losses import MarginPCL
            criterion_pcl = MarginPCL(
                margin_pos=args.pcl_margin_pos,
                margin_neg=args.pcl_margin_neg,
                hard_scale=args.pcl_hard_scale,
                use_dynamic_hardness=True
            )
            logger.info(f"📌 [P1+P2] Using Margin PCL (margin_pos={args.pcl_margin_pos}, margin_neg={args.pcl_margin_neg}, scale={args.pcl_hard_scale}, weight={args.pcl_weight})")
        else:  # v4 default
            from model.losses import HardLabelWeightedPCL
            criterion_pcl = HardLabelWeightedPCL(
                hard_weight=args.pcl_hard_weight
            )
            logger.info(f"📌 [B4-v4] Using Static Weighted PCL (hard_weight={args.pcl_hard_weight}, pcl_weight={args.pcl_weight})")

    # === P3: Phoneme-Position PCL 初始化 ===
    criterion_phon_pcl = None
    if args.use_phoneme_pcl:
        from model.losses import PhonemePositionPCL
        criterion_phon_pcl = PhonemePositionPCL(
            gamma=args.phoneme_pcl_gamma,
            margin_neg=args.phoneme_pcl_margin
        )
        logger.info(f"📌 [P3] Using Phoneme-Position PCL (gamma={args.phoneme_pcl_gamma}, margin={args.phoneme_pcl_margin}, weight={args.phoneme_pcl_weight})")

    # === AsymmetricMinPCL 初始化 ===
    criterion_neg_min_pcl = None
    if args.use_neg_min_pcl:
        from model.losses import AsymmetricMinPCL
        criterion_neg_min_pcl = AsymmetricMinPCL(
            margin_neg=args.neg_min_pcl_margin,
            margin_pos=args.neg_min_pcl_margin_pos,
            hard_scale=args.neg_min_pcl_hard_scale,
            enable_pos_floor=not args.neg_min_pcl_no_pos_floor,
        )
        logger.info(
            f"📌 [AsymMinPCL] margin_neg={args.neg_min_pcl_margin}, "
            f"margin_pos={args.neg_min_pcl_margin_pos}, "
            f"hard_scale={args.neg_min_pcl_hard_scale}, "
            f"weight={args.neg_min_pcl_weight}, "
            f"pos_floor={'ON' if not args.neg_min_pcl_no_pos_floor else 'OFF'}"
        )

    # === MFA Auxiliary CE Loss 初始化 ===
    criterion_aux_ce = None
    if args.enable_aux_ce:
        from model.losses import AuxCELoss
        criterion_aux_ce = AuxCELoss(
            n_phonemes=args.n_phonemes,
            ignore_index=0,  # PAD
            alpha_agc=args.alpha_agc,
            label_smoothing=args.aux_label_smoothing
        )
        logger.info(f"📌 [MFA Aux CE] Enabled with λ_aux={args.lambda_aux}, α_agc={args.alpha_agc}, label_smoothing={args.aux_label_smoothing}, n_phonemes={args.n_phonemes}")
    
    # === Score-level supervision 啟動 log ===
    if args.use_score_loss:
        logger.info(f"📌 [L_score] Enabled with weight={args.score_loss_weight}")
    
    # === 載入 G2P → MFA phoneme mapping (AGC 需要) ===
    g2p_to_mfa_tensor = None
    if args.enable_aux_ce and args.alpha_agc > 0 and args.frame_labels_path:
        # 動態從 train_dataloader 的 dataset 獲取 phoneme 列表，確保 index 順序一致
        dataset = train_dataloader.dataset
        if hasattr(dataset, 'idx2p'):
            g2p_phonemes = [dataset.idx2p[i] for i in range(len(dataset.idx2p))]
            logger.info(f"📌 [AGC] Using phonemes from dataset.idx2p (len={len(g2p_phonemes)})")
        else:
            g2p_phonemes = get_g2p_phonemes()
            logger.info(f"📌 [AGC] Using hardcoded get_g2p_phonemes() (len={len(g2p_phonemes)})")
        
        # 載入 MFA vocab
        mfa_vocab = load_mfa_vocab(args.frame_labels_path)
        logger.info(f"📌 [AGC] MFA vocab loaded: {len(mfa_vocab)} phonemes")
        
        # 建立 mapping
        g2p_to_mfa_tensor = build_g2p_to_mfa_mapping(g2p_phonemes, mfa_vocab)
        g2p_to_mfa_tensor = g2p_to_mfa_tensor.to(accelerator.device)
        
        # 驗證映射：檢查常見音素
        logger.info(f"📌 [AGC] G2P→MFA mapping verification:")
        test_phonemes = [(0, '<pad>'), (1, 'AA0'), (19, 'B'), (44, 'M')]
        for idx, name in test_phonemes:
            if idx < len(g2p_to_mfa_tensor):
                mapped_idx = g2p_to_mfa_tensor[idx].item()
                logger.info(f"    G2P[{idx}]='{name}' → MFA[{mapped_idx}]")

    # Metrics
    train_loss = MeanMetric()
    train_loss_d = MeanMetric()
    train_loss_sce = MeanMetric()
    train_loss_aux = MeanMetric()  # MFA Aux CE loss tracking
    test_loss = MeanMetric()
    test_loss_d = MeanMetric()
    train_auc = BinaryAUROC()  # Combined AUC (P_utt × P_spk vs z_final)
    train_eer = eer()
    # NOTE: test_auc, test_eer removed - use PersonalizedKWSMetrics instead for proper C-KWS/TB-KWS/TO-KWS metrics
    
    # === NEW: 真正的 C-KWS 指標 (P_utt vs z_keyword) ===
    train_ckws_auc = BinaryAUROC()
    train_ckws_eer = eer()


    # === 根據參數選擇 Optimizer ===
    def create_optimizer(params, lr, args):
        """Create optimizer based on command-line arguments"""
        if args.optimizer_type == 'adam':
            # Baseline: 原版設定，無 weight_decay
            return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-7)
        elif args.optimizer_type == 'adam_wd':
            # Adam + weight_decay (L2 regularization)
            return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-7, weight_decay=args.weight_decay)
        elif args.optimizer_type == 'adamw':
            # AdamW: Decoupled weight decay
            return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-7, weight_decay=args.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer_type: {args.optimizer_type}")

    # Optimizer (only trainable parameters)
    # === v3.0: FiLM 分組學習率 ===
    if args.personalized:
        # 取得參數分組
        unwrapped_model = accelerator.unwrap_model(model)
        film_params = unwrapped_model.get_film_parameters()
        film_params_set = set(film_params)
        
        # === Calibration 參數分離 ===
        calib_params = []
        if hasattr(unwrapped_model, 'spk_scale') and isinstance(unwrapped_model.spk_scale, nn.Parameter):
            calib_params.append(unwrapped_model.spk_scale)
        if hasattr(unwrapped_model, 'spk_bias') and isinstance(unwrapped_model.spk_bias, nn.Parameter):
            calib_params.append(unwrapped_model.spk_bias)
        calib_params_set = set(calib_params)
        
        # 其餘參數（排除 FiLM 和 Calibration）
        other_params = [
            p for p in unwrapped_model.parameters()
            if p.requires_grad
            and p not in film_params_set
            and p not in calib_params_set
        ]
        
        # 學習率設定
        film_lr = args.lr * args.film_lr_mult
        calib_lr = args.lr * args.calib_lr_mult
        
        param_groups = [
            {'params': other_params, 'lr': args.lr},
            {'params': film_params, 'lr': film_lr, 'name': 'film'},
        ]
        if calib_params:
            param_groups.append({'params': calib_params, 'lr': calib_lr, 'name': 'calibration'})
        
        optimizer = create_optimizer(param_groups, args.lr, args)
        logger.info(f"📌 FiLM LR: {film_lr} ({args.film_lr_mult}x), Others LR: {args.lr}")
        if calib_params:
            logger.info(f"📌 Calibration LR: {calib_lr} ({args.calib_lr_mult}x) - {len(calib_params)} params")
    else:
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = create_optimizer(list(trainable_params), args.lr, args)
    
    # Log optimizer type
    if args.optimizer_type == 'adam':
        logger.info("📌 Using Adam optimizer (no weight_decay) - BASELINE")
    elif args.optimizer_type == 'adam_wd':
        logger.info(f"📌 Using Adam optimizer with weight_decay={args.weight_decay}")
    elif args.optimizer_type == 'adamw':
        logger.info(f"📌 Using AdamW optimizer with weight_decay={args.weight_decay}")

    # === 根據參數選擇 Scheduler ===
    scheduler = None
    if args.scheduler_type == 'none':
        logger.info("📌 No scheduler - BASELINE")
    elif args.scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        logger.info("📌 Using ReduceLROnPlateau scheduler")
    elif args.scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.epoch, 
            eta_min=1e-6
        )
        logger.info(f"📌 Using CosineAnnealingLR scheduler (T_max={args.epoch})")
    else:
        raise ValueError(f"Unknown scheduler_type: {args.scheduler_type}")

    # Prepare for distributed training
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    for i in range(len(eval_dataloader)):
        eval_dataloader[i] = accelerator.prepare(eval_dataloader[i])

    loss_object, loss_object_sce, train_loss, train_loss_d, train_loss_sce, \
    test_loss, test_loss_d, train_auc, train_eer, \
    train_ckws_auc, train_ckws_eer = \
        accelerator.prepare(
            loss_object, loss_object_sce, train_loss, train_loss_d,
            train_loss_sce, test_loss, test_loss_d, train_auc, train_eer,
            train_ckws_auc, train_ckws_eer
        )

    if accelerator.is_main_process:
        accelerator.init_trackers("logs")

    # Training info
    logger.info("***** Running P-PhonMatchNet Training *****")
    logger.info(f"  Mode: {args.mode}")
    logger.info(f"  Personalized: {args.personalized}")
    logger.info(f"  Epochs: {args.epoch}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    if args.personalized:
        logger.info(f"  Using A+C scheme with sample-level masking")

    # Log config
    if accelerator.is_main_process:
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                for arg in vars(args):
                    tracker.writer.add_text("config/" + str(arg), str(getattr(args, arg)))

    global_step = 0
    first_epoch = 0

    # Early stopping variables
    best_val_eer = float('inf')  # Track best EER for primary mode
    best_metrics = None  # Store all mode metrics at best epoch
    patience_counter = 0
    patience = 10  # Early stopping patience

    # Resume from checkpoint if available
    output_path = Path(args.output_dir)
    potential_ckpts = list(output_path.glob("checkpoint_*.pth"))

    if potential_ckpts:
        latest_ckpt = max(potential_ckpts, key=lambda p: int(p.stem.split('_')[1]))
        logger.info(f"🔄 Found checkpoint: {latest_ckpt}")
        logger.info("🔄 Resuming training...")

        checkpoint = torch.load(latest_ckpt, map_location='cpu')

        unwrap_model = accelerator.unwrap_model(model)
        unwrap_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        first_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['step']

        # Restore early stopping state
        best_val_eer = checkpoint.get('best_val_eer', float('inf'))
        best_metrics = checkpoint.get('best_metrics', None)
        patience_counter = checkpoint.get('patience_counter', 0)

        logger.info(f"👉 Resuming from Epoch {first_epoch}, Step {global_step}")
        logger.info(f"📊 Best EER so far: {best_val_eer*100:.2f}%, Patience: {patience_counter}/{patience}")
    else:
        logger.info("ℹ️ No checkpoint found. Starting from scratch.")
        
        # === 初始化 validation_log.txt 並寫入實驗配置 ===
        if accelerator.is_main_process:
            val_log_path = os.path.join(args.output_dir, "validation_log.txt")
            with open(val_log_path, 'w') as f:  # 新訓練用 write mode 覆寫
                f.write("=" * 70 + "\n")
                f.write("P-PhonMatchNet Training - Experiment Configuration\n")
                f.write("=" * 70 + "\n")
                f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Output Dir: {args.output_dir}\n")
                f.write("\n")
                
                # === Core Settings ===
                f.write("[Core Settings]\n")
                f.write(f"  Mode: {args.mode}\n")
                f.write(f"  Personalized: {args.personalized}\n")
                f.write(f"  Epochs: {args.epoch}\n")
                f.write(f"  Batch Size: {args.batch_size}\n")
                f.write(f"  Learning Rate: {args.lr}\n")
                f.write("\n")
                
                # === Model Architecture ===
                f.write("[Model Architecture]\n")
                f.write(f"  Audio Input: {args.audio_input}\n")
                f.write(f"  Text Input: {args.text_input}\n")
                f.write(f"  Stack Extractor: {args.stack_extractor}\n")
                f.write("\n")
                
                # === Ablation Study Settings ===
                f.write("[Ablation Study Settings]\n")
                f.write(f"  Disable FiLM: {getattr(args, 'disable_film', False)}\n")
                f.write(f"  Disable SV Branch: {getattr(args, 'disable_sv_branch', False)}\n")
                f.write(f"  Unfreeze Speaker Encoder: {getattr(args, 'unfreeze_speaker_encoder', False)}\n")
                f.write(f"  Finetuned Speaker Encoder: {getattr(args, 'finetuned_speaker_encoder', None)}\n")
                f.write(f"  Enrollment Dropout: {getattr(args, 'enrollment_dropout', 0.0)}\n")
                f.write("\n")
                
                # === Optimizer & Scheduler ===
                f.write("[Optimizer & Scheduler]\n")
                f.write(f"  Optimizer: {args.optimizer_type}\n")
                f.write(f"  Scheduler: {args.scheduler_type}\n")
                f.write(f"  Weight Decay: {getattr(args, 'weight_decay', 0)}\n")
                f.write("\n")
                
                # === Dataset ===
                f.write("[Dataset]\n")
                f.write(f"  Train PKL: {args.train_pkl}\n")
                f.write(f"  Speaker Ratio: {getattr(args, 'speaker_ratio', 0.5)}\n")
                f.write("\n")
                
                f.write("=" * 70 + "\n")
                f.write("Validation Results:\n")
                f.write("=" * 70 + "\n")
            
            logger.info(f"📝 Created validation_log.txt with experiment config")

    # === Initialize extended test loaders for GSC and Qualcomm ===
    extended_loaders = None
    extended_metrics_results = {}  # Store GSC/Qualcomm metrics for logging
    if args.eval_gsc_qualcomm and accelerator.is_main_process:
        logger.info("\n📊 Initializing extended test loaders (GSC + Qualcomm)...")
        train_args_dict = vars(args)  # Current args as dict
        extended_loaders = create_extended_test_loaders(args, train_args_dict)
        logger.info("✅ Extended loaders ready")

    # Training loop
    for epoch in range(first_epoch, args.epoch):
        model.train()
        
        # === A6: Curriculum Gembed — update epoch ===
        if getattr(args, 'gemb_curriculum', False):
            raw_model = accelerator.unwrap_model(model)
            alpha = raw_model.set_current_epoch(epoch)
            logger.info(f"📈 [Curriculum] Epoch {epoch}: gembed α = {alpha:.3f}")
        
        # === v5.0: 移除 Dynamic Scale Curriculum ===
        # spk_scale 由模型自行學習，不再需要手動限制上限
        
        progress_bar = tqdm(
            range(int(args.epoch * train_len / (args.batch_size * accelerator.num_processes))),
            initial=global_step,
            desc=f"Epoch {epoch}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
        )

        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            # Prepare inputs based on audio type
            if args.audio_input == "raw":
                if args.audio_noise:
                    speech_input = batch["x_noisy"]
                else:
                    speech_input = batch["x"]
                speech_len = batch["x_len"]
            elif args.audio_input == "google_embed":
                speech_input = batch["gemb"]
                speech_len = batch["gemb_len"]
            elif args.audio_input in ("both", "enhanced_gembed"):
                if args.audio_noise:
                    speech_input = (batch["x_noisy"], batch["gemb"])
                else:
                    speech_input = (batch["x"], batch["gemb"])
                speech_len = (batch["x_len"], batch["gemb_len"])
            else:
                raise NotImplementedError

            # Text input
            text_input = batch["y"]
            text_len = batch["y_len"]

            # Forward pass
            # NOTE: args.mode does NOT affect forward pass - only used for early stopping
            if args.personalized:
                # === v3.0: 從 batch 取得 same_speaker ===
                # speaker_label: [B, 1]，1 = same speaker，0 = different speaker
                same_speaker = batch["speaker_label"].squeeze(-1).bool()  # [B]
                
                # Personalized: pass same_speaker for conditional FiLM
                output = model(
                    speech=speech_input,
                    text=text_input,
                    speech_len=speech_len,
                    text_len=text_len,
                    enrollment_audio=batch["enrollment_audio"],
                    same_speaker=same_speaker,  # v3.0: conditional FiLM
                    raw_audio_for_spk=batch.get("x_noisy", batch["x"]) if getattr(args, "audio_noise", False) else batch.get("x"),
                )
            else:
                # Standard PhonMatchNet: no enrollment
                output = model(
                    speech=speech_input,
                    text=text_input,
                    speech_len=speech_len,
                    text_len=text_len,
                )

            # Extract outputs
            P_utt = output['P_utt']      # 純 keyword 分數
            P_spk = output['P_spk']      # Speaker 分數
            score = output['score']      # P_utt × P_spk
            
            seq_logit = output['seq_ce_logit']  # [B, T_t] phoneme logits
            seq_logit_mask = output['seq_ce_logit_mask']

            # === CRITICAL FIX: 分離 Loss 計算 ===
            if args.personalized:
                # === Loss 1: Keyword-only Loss (for C-KWS capability) ===
                # 使用 z_keyword 和 LD (logit)，確保純 keyword 判別能力
                # NOTE: BCEWithLogitsLoss 需要 logit (sigmoid 前)，不是 probability
                LD_logit = output['LD']  # 這是 sigmoid 前的 logit
                
                # === Focal Loss 支援 (B2/B3 Ablation Study) ===
                if args.focal_loss_phon or args.focal_loss_utt:
                    # 分離計算 L_utt 和 L_phon
                    from criterion.total import detection_loss, sequence_cross_entropy
                    
                    # L_utt (LD)
                    if args.focal_loss_utt:
                        # B3: 使用 Focal Loss for L_utt
                        LD_kws = criterion_utt_focal(P_utt.squeeze(-1), batch['z_keyword'].squeeze(-1))
                    else:
                        # 原本的 BCE
                        LD_kws = detection_loss(batch['z_keyword'], LD_logit, reduction='sum')
                    
                    # L_phon (LC)
                    if args.focal_loss_phon:
                        # B2: 使用 Focal Loss for L_phon（修正版：使用 per-phoneme paired labels）
                        LC_kws = criterion_phon_focal(
                            speech_label=batch['l'],
                            text_label=batch['t'],
                            logits=seq_logit,
                            logits_mask=seq_logit_mask
                        )
                    else:
                        # 原本的 sequence_cross_entropy
                        LC_kws = sequence_cross_entropy(
                            batch['l'], batch['t'], seq_logit, seq_logit_mask, reduction='sum'
                        )
                    
                    # 組合 loss_kws
                    loss_kws = args.loss_weight[0] * LD_kws + args.loss_weight[1] * LC_kws
                else:
                    # 原本的 TotalLoss_SCE
                    loss_kws, LD_kws, LC_kws = loss_object_sce(
                        batch['z_keyword'],  # 純 keyword 標籤
                        LD_logit,            # 使用 logit 而非 P_utt (probability)
                        batch['l'],
                        batch['t'],
                        seq_logit,
                        seq_logit_mask
                    )
                
                # === ABLATION FIX: 當 disable_sv_branch 時，只用 loss_kws ===
                # 避免 z_final (含 z_speaker) 污染 keyword matching 能力
                if args.disable_sv_branch:
                    loss = loss_kws
                    LD = LD_kws
                    LC = LC_kws
                    w_kws = 1.0
                    w_personal = 0.0
                    loss_personal = torch.tensor(0.0, device=loss_kws.device)
                    LD_personal = torch.tensor(0.0, device=loss_kws.device)
                else:
                    # === v5.0: L_personal 監督 P_spk ===
                    # 所有樣本都參與訓練，讓 P_spk 學習正確的 speaker verification
                    # P_spk 應該匹配 z_speaker (speaker_label)
                    P_spk_clamped = P_spk.squeeze(-1).clamp(1e-7, 1 - 1e-7)  # [B]
                    z_speaker_float = batch['speaker_label'].squeeze(-1).float()  # [B]
                    loss_personal = criterion_bce(P_spk_clamped, z_speaker_float).sum()  # scalar (sum to match loss_kws)
                    
                    # LD_personal 沿用（為了與後續代碼相容）
                    LD_personal = loss_personal
                    
                    # === v5.0: 固定權重，移除課程學習 ===
                    w_kws = 1.0
                    w_personal = args.personal_loss_weight
                
                # 組合 Loss（僅在非 ablation 模式時）
                if not args.disable_sv_branch:
                    loss = w_kws * loss_kws + w_personal * loss_personal
                    LD = w_kws * LD_kws + w_personal * LD_personal
                    LC = LC_kws  # L_phon 只需計算一次
                
                # === L_score: Content-aware anti-drift (新增) ===
                L_score = torch.tensor(0.0, device=loss.device)
                if args.use_score_loss and not args.disable_sv_branch:
                    # Use same fusion as model for L_score
                    unwrapped = accelerator.unwrap_model(model)
                    Score_supervised = unwrapped.compute_score(
                        P_utt.detach().squeeze(-1),  # detach P_utt
                        P_spk.squeeze(-1)
                    )
                    z_joint = (batch['z_keyword'].squeeze(-1).float() * batch['speaker_label'].squeeze(-1).float()).to(loss.device)
                    L_score = F.binary_cross_entropy(
                        Score_supervised.clamp(1e-7, 1 - 1e-7),
                        z_joint,
                        reduction='sum'  # sum to match loss_kws, 全局 /= batch_size 會統一正規化
                    )
                    loss = loss + args.score_loss_weight * L_score
                
                # 用於 metrics 更新的 prob（保持向後相容）
                prob = score
            else:
                # Standard mode: 只用 keyword loss
                prob = P_utt
                LD_logit = output['LD']  # 使用 logit 而非 probability
                
                # === Focal Loss 支援 (B2/B3 Ablation Study) ===
                if args.focal_loss_phon or args.focal_loss_utt:
                    # 分離計算 L_utt 和 L_phon
                    from criterion.total import detection_loss, sequence_cross_entropy
                    
                    # L_utt (LD)
                    if args.focal_loss_utt:
                        # B3: 使用 Focal Loss for L_utt
                        LD = criterion_utt_focal(P_utt.squeeze(-1), batch['z'].squeeze(-1))
                    else:
                        # 原本的 BCE
                        LD = detection_loss(batch['z'], LD_logit, reduction='sum')
                    
                    # L_phon (LC)
                    if args.focal_loss_phon:
                        # B2: 使用 Focal Loss for L_phon（修正版：使用 per-phoneme paired labels）
                        LC = criterion_phon_focal(
                            speech_label=batch['l'],
                            text_label=batch['t'],
                            logits=seq_logit,
                            logits_mask=seq_logit_mask
                        )
                    else:
                        # 原本的 sequence_cross_entropy
                        LC = sequence_cross_entropy(
                            batch['l'], batch['t'], seq_logit, seq_logit_mask, reduction='sum'
                        )
                    
                    # 組合 total loss
                    loss = args.loss_weight[0] * LD + args.loss_weight[1] * LC
                else:
                    # 原本的 TotalLoss_SCE
                    loss, LD, LC = loss_object_sce(
                        batch['z'],
                        LD_logit,            # 使用 logit 而非 prob
                        batch['l'],
                        batch['t'],
                        seq_logit,
                        seq_logit_mask
                    )

            loss /= args.batch_size
            LD /= args.batch_size
            LC /= args.batch_size

            # === Weighted PCL Loss (B4 + P1/P2 Improvements) ===
            L_pcl = torch.tensor(0.0, device=loss.device)
            if args.use_weighted_pcl and criterion_pcl is not None:
                P_phon = torch.sigmoid(seq_logit)  # [B, T_t]
                kw_label = batch['z_keyword'] if args.personalized else batch['z']
                
                if args.pcl_version in ['dynamic', 'margin']:
                    # P1/P2: 需要 P_utt 做 dynamic hardness weighting
                    L_pcl = criterion_pcl(
                        P_phon=P_phon,
                        P_utt=P_utt.detach(),  # detach 避免 PCL 梯度影響 P_utt head
                        speech_label=batch['l'],
                        text_label=batch['t'],
                        z_keyword=kw_label,
                        phoneme_mask=seq_logit_mask
                    )
                else:
                    # v4: 需要 is_hard（從 dataset type 判斷）
                    if 'type' in batch:
                        is_hard_list = ['hard' in str(t) for t in batch['type']]
                        is_hard = torch.tensor(is_hard_list, dtype=torch.bool, device=loss.device)
                    else:
                        is_hard = torch.zeros(P_utt.size(0), dtype=torch.bool, device=loss.device)
                    L_pcl = criterion_pcl(
                        P_phon=P_phon,
                        speech_label=batch['l'],
                        text_label=batch['t'],
                        z_keyword=kw_label,
                        is_hard=is_hard,
                        phoneme_mask=seq_logit_mask
                    )
                L_pcl /= args.batch_size  # 與其他 loss 一致的縮放
                loss = loss + args.pcl_weight * L_pcl

            # === P3: Phoneme-Position PCL ===
            L_phon_pcl = torch.tensor(0.0, device=loss.device)
            if args.use_phoneme_pcl and criterion_phon_pcl is not None:
                P_phon = torch.sigmoid(seq_logit)
                kw_label = batch['z_keyword'] if args.personalized else batch['z']
                L_phon_pcl = criterion_phon_pcl(
                    P_phon=P_phon,
                    speech_label=batch['l'],
                    text_label=batch['t'],
                    z_keyword=kw_label,
                    phoneme_mask=seq_logit_mask
                )
                L_phon_pcl /= args.batch_size
                loss = loss + args.phoneme_pcl_weight * L_phon_pcl

            # === NegMinPCL ===
            L_neg_min_pcl = torch.tensor(0.0, device=loss.device)
            if args.use_neg_min_pcl and criterion_neg_min_pcl is not None:
                P_phon_sig = torch.sigmoid(seq_logit)            # [B, T_t]
                kw_label = batch['z_keyword'] if args.personalized else batch['z']

                L_neg_min_pcl = criterion_neg_min_pcl(
                    P_phon=P_phon_sig,
                    P_utt=P_utt,                                 # 內部會 detach
                    z_keyword=kw_label,
                    phoneme_mask=seq_logit_mask
                )
                loss = loss + args.neg_min_pcl_weight * L_neg_min_pcl

            # === MFA Auxiliary CE Loss (A4: frame-level AGC) ===
            L_aux = torch.tensor(0.0, device=loss.device)
            aux_accuracy = 0.0
            aux_acc_active = 0.0  # 非靜音準確率
            n_mismatch = 0
            
            if args.enable_aux_ce and criterion_aux_ce is not None:
                aux_logits = output.get('aux_logits')
                frame_labels = batch.get('frame_labels')
                
                if aux_logits is not None and frame_labels is not None:
                    # 計算 frame-level mismatch mask（只在 alpha_agc > 0 時啟用）
                    mismatch_mask = None
                    if args.alpha_agc > 0 and g2p_to_mfa_tensor is not None:
                        kw_label = batch['z_keyword'] if args.personalized else batch['z']
                        mismatch_mask = compute_frame_mismatch_mask(
                            frame_labels=frame_labels,
                            text_indices=batch['t'],
                            text_lens=batch['t_len'],
                            g2p_to_mfa=g2p_to_mfa_tensor,
                            z_keyword=kw_label,
                        )
                        
                        # === DEBUG: 每 100 steps 輸出 mismatch 統計 ===
                        if global_step % 100 == 0 and accelerator.is_main_process:
                            total_frames = mismatch_mask.numel()
                            mismatch_frames = mismatch_mask.sum().item()
                            mismatch_ratio = mismatch_frames / (total_frames + 1e-8)
                            
                            # 檢查 negative samples
                            z_squeezed = kw_label.squeeze()
                            is_neg = (z_squeezed == 0)
                            n_neg = is_neg.sum().item()
                            
                            logger.info(f"\n[DEBUG AGC Step {global_step}]")
                            logger.info(f"  - Batch size: {frame_labels.shape[0]}, Frames/sample: {frame_labels.shape[1]}")
                            logger.info(f"  - Negative samples: {n_neg}/{frame_labels.shape[0]}")
                            logger.info(f"  - Mismatch ratio: {mismatch_ratio:.2%} ({int(mismatch_frames)}/{total_frames})")
                            
                            # 檢查 text mapping（抽第一個 negative sample）
                            if n_neg > 0:
                                neg_idx = is_neg.nonzero(as_tuple=True)[0][0].item()
                                text_sample = batch['t'][neg_idx][:5].long()
                                mapped_sample = g2p_to_mfa_tensor[text_sample.clamp(0, len(g2p_to_mfa_tensor)-1)]
                                frame_sample = frame_labels[neg_idx][:5]
                                logger.info(f"  - Sample text[0:5]: {text_sample.tolist()} → mapped: {mapped_sample.tolist()}")
                                logger.info(f"  - Sample frame_labels[0:5]: {frame_sample.tolist()}")
                            
                            if mismatch_frames == 0:
                                logger.warning("  ⚠️ WARNING: L_agc is 0! Mismatch mask is empty.")
                    
                    aux_result = criterion_aux_ce(
                        aux_logits=aux_logits,
                        frame_labels=frame_labels,
                        mismatch_mask=mismatch_mask,
                    )
                    L_aux = aux_result['loss']
                    aux_accuracy = aux_result['accuracy'].item()
                    aux_acc_active = aux_result.get('acc_active', torch.tensor(0.0)).item()
                    n_mismatch = aux_result.get('n_mismatch', 0)
                    loss = loss + args.lambda_aux * L_aux

            # Update metrics
            train_auc.update(prob.detach(), batch['z'].detach())  # Combined/TO-KWS
            train_eer.update(batch['z'].detach(), prob.detach())
            
            # === NEW: 更新真正的 C-KWS 指標 ===
            if args.personalized:
                train_ckws_auc.update(output['P_utt'].detach(), batch['z_keyword'].detach())
                train_ckws_eer.update(batch['z_keyword'].detach(), output['P_utt'].detach())

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update loss metrics
            train_loss.update(loss.item())
            train_loss_d.update(LD.item())
            train_loss_sce.update(LC.item())
            if args.enable_aux_ce:
                train_loss_aux.update(L_aux.item() if isinstance(L_aux, torch.Tensor) else L_aux)

            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Prepare logging dict
                log_dict = {
                    "epoch": epoch,
                    "train/loss/total": train_loss.compute().detach().item(),
                    "train/loss/d": train_loss_d.compute().detach().item(),
                    "train/loss/sce": train_loss_sce.compute().detach().item(),
                    "train/combined_auc": train_auc.compute().detach().item(),  # 重命名：P_utt × P_spk vs z_final
                    "train/combined_eer": train_eer.compute().detach().item(),  # 重命名
                }
                
                # === NEW: 新增真正的 C-KWS 指標 ===
                if args.personalized:
                    log_dict.update({
                        "train/c_kws_auc": train_ckws_auc.compute().detach().item(),  # 真正的 C-KWS: P_utt vs z_keyword
                        "train/c_kws_eer": train_ckws_eer.compute().detach().item(),
                    })
                
                # === MFA Aux CE logging ===
                if args.enable_aux_ce:
                    log_dict.update({
                        "train/aux_ce/loss": train_loss_aux.compute().detach().item(),
                        "train/aux_ce/accuracy": aux_accuracy,
                        "train/aux_ce/acc_active": aux_acc_active,  # 非靜音準確率
                        "train/agc/n_mismatch_frames": n_mismatch,  # A4: frame-level AGC monitoring
                    })
                
                # === AsymmetricMinPCL logging ===
                if args.use_neg_min_pcl:
                    neg_min_pcl_log = {
                        "train/asym_pcl/loss": L_neg_min_pcl.item() if isinstance(L_neg_min_pcl, torch.Tensor) else L_neg_min_pcl,
                    }
                    with torch.no_grad():
                        P_phon_sig = torch.sigmoid(seq_logit)
                        P_phon_masked = P_phon_sig.clone()
                        P_phon_masked[~seq_logit_mask] = 1.0
                        s_all = P_phon_masked.min(dim=-1).values  # [B]

                        kw_label = batch['z_keyword'] if args.personalized else batch['z']
                        is_neg = (kw_label.view(-1) == 0)
                        is_pos = (kw_label.view(-1) == 1)

                        if is_neg.any():
                            s_neg = s_all[is_neg]
                            neg_min_pcl_log.update({
                                "train/asym_pcl/neg_s_mean": s_neg.mean().item(),
                                "train/asym_pcl/neg_above_margin": (s_neg > args.neg_min_pcl_margin).float().mean().item(),
                                "train/asym_pcl/weight_mean": (1.0 + args.neg_min_pcl_hard_scale * P_utt.detach().view(-1)[is_neg]).mean().item(),
                            })

                        if is_pos.any():
                            s_pos = s_all[is_pos]
                            neg_min_pcl_log.update({
                                "train/asym_pcl/pos_s_mean": s_pos.mean().item(),
                                "train/asym_pcl/pos_s_min": s_pos.min().item(),
                                "train/asym_pcl/pos_below_floor": (s_pos < args.neg_min_pcl_margin_pos).float().mean().item(),
                            })
                    log_dict.update(neg_min_pcl_log)
                
                # === A5-Fix + A6: Feature / Curriculum / Gate monitoring ===
                if args.gemb_drop_rate > 0 or getattr(args, 'gemb_curriculum', False) or getattr(args, 'stream_fusion', 'add') == 'gated':
                    log_dict.update({
                        "train/feature/LDN_magnitude": output.get('LDN_magnitude', 0.0),
                        "train/feature/gembed_magnitude": output.get('gembed_magnitude', 0.0),
                        "train/feature/energy_ratio": output.get('LDN_magnitude', 0.0) / (output.get('gembed_magnitude', 1e-8) + 1e-8),
                    })
                    if getattr(args, 'gemb_curriculum', False):
                        log_dict["train/curriculum/gembed_alpha"] = output.get('gembed_alpha', 1.0)
                    if args.gemb_drop_rate > 0:
                        log_dict["train/feature/gemb_drop_rate"] = args.gemb_drop_rate
                    # A6: Gated Fusion gate stats
                    if getattr(args, 'stream_fusion', 'add') == 'gated':
                        log_dict["train/gate/mean"] = output.get('gembed_alpha', 0.0)
                        # Get detailed gate stats from model
                        unwrap = accelerator.unwrap_model(model)
                        if hasattr(unwrap, '_last_ldn_gate') and unwrap._last_ldn_gate is not None:
                            _g = unwrap._last_ldn_gate
                            log_dict["train/gate/std"] = _g.std().item()
                            log_dict["train/gate/min"] = _g.min().item()
                            log_dict["train/gate/max"] = _g.max().item()

                # === NEW: Add calibration parameters for personalized mode ===
                if args.personalized:
                    # Get unwrapped model (in case using DDP/FSDP)
                    unwrapped_model = accelerator.unwrap_model(model)

                    # Log calibration parameters (only if they exist and are learnable)
                    if hasattr(unwrapped_model, 'spk_scale') and isinstance(unwrapped_model.spk_scale, nn.Parameter):
                        log_dict["train/spk_scale"] = unwrapped_model.spk_scale.item()
                    if hasattr(unwrapped_model, 'spk_bias') and isinstance(unwrapped_model.spk_bias, nn.Parameter):
                        log_dict["train/spk_bias"] = unwrapped_model.spk_bias.item()
                    
                    # Log calibration mode
                    cal_mode = getattr(unwrapped_model, 'calibration_mode', 'full')
                    
                    # P_spk distribution logging (all calibration modes)
                    with torch.no_grad():
                        P_spk_flat = P_spk.squeeze(-1) if P_spk.dim() > 1 else P_spk
                        z_spk = batch.get('speaker_label', batch.get('z_spk'))
                        if z_spk is not None:
                            z_spk_flat = z_spk.squeeze(-1) if z_spk.dim() > 1 else z_spk
                            is_same_spk = (z_spk_flat == 1)
                            is_diff_spk = (z_spk_flat == 0)
                            if is_same_spk.any():
                                log_dict["calib/P_spk_same_mean"] = P_spk_flat[is_same_spk].mean().item()
                            if is_diff_spk.any():
                                log_dict["calib/P_spk_diff_mean"] = P_spk_flat[is_diff_spk].mean().item()
                    
                    # === v5.0: 簡化的 loss 監控（移除課程學習相關）===
                    if 'loss_kws' in locals():
                        log_dict.update({
                            "train/loss_kws": loss_kws.item() / args.batch_size,
                            "train/loss_personal": loss_personal.item() / args.batch_size if hasattr(loss_personal, 'item') else 0.0,
                        })
                    
                    # === Log FiLM stats ===
                    if hasattr(unwrapped_model, 'get_film_stats'):
                        film_stats = unwrapped_model.get_film_stats()
                        
                        log_dict.update({
                            # FiLM modulation 統計
                            "train/film_gamma_mean": film_stats.get("gamma_mean", 0),
                            "train/film_gamma_std": film_stats.get("gamma_std", 0),
                            "train/film_beta_mean": film_stats.get("beta_mean", 0),
                            "train/film_beta_std": film_stats.get("beta_std", 0),
                        })
                    
                    # === v5.2: Log Gate stats ===
                    if hasattr(unwrapped_model, 'get_gate_stats'):
                        gate_stats = unwrapped_model.get_gate_stats()
                        if gate_stats is not None:
                            log_dict.update({
                                "train/gate_mean": gate_stats.get("gate_mean", 0),
                                "train/gate_std": gate_stats.get("gate_std", 0),
                                "train/gate_min": gate_stats.get("gate_min", 0),
                                "train/gate_max": gate_stats.get("gate_max", 0),
                            })

                    # === Score-level supervision logging ===
                    if args.use_score_loss:
                        log_dict["train/loss/L_score"] = L_score.item() / args.batch_size
                        with torch.no_grad():
                            Score_val = unwrapped_model.compute_score(
                                P_utt.squeeze(-1), P_spk.squeeze(-1)
                            )
                            z_joint_log = batch['z_keyword'].squeeze(-1).float() * batch['speaker_label'].squeeze(-1).float()
                            pos_mask = (z_joint_log == 1)
                            neg_mask = (z_joint_log == 0)
                            if pos_mask.any():
                                log_dict["train/score_sup/pos_mean"] = Score_val[pos_mask].mean().item()
                            if neg_mask.any():
                                log_dict["train/score_sup/neg_mean"] = Score_val[neg_mask].mean().item()
                            # 最危險 case: imposter + 正確關鍵詞
                            kw_match = (batch['z_keyword'].squeeze(-1) == 1)
                            spk_mismatch = (batch['speaker_label'].squeeze(-1) == 0)
                            imposter_correct_kw = kw_match & spk_mismatch
                            if imposter_correct_kw.any():
                                log_dict["train/score_sup/imposter_correct_kw_score"] = Score_val[imposter_correct_kw].mean().item()
                                log_dict["train/score_sup/imposter_correct_kw_P_spk"] = P_spk.squeeze(-1)[imposter_correct_kw].mean().item()

                accelerator.log(log_dict, step=global_step)

                logs = {
                    "train_loss": train_loss.compute().detach().item(),
                    "train_loss_d": train_loss_d.compute().detach().item(),
                    "train_loss_sce": train_loss_sce.compute().detach().item(),
                }
                progress_bar.set_postfix(**logs)

        # End of epoch
        logger.info(f"Epoch {epoch} - Train AUC: {train_auc.compute().detach().item():.4f}")
        logger.info(f"Epoch {epoch} - Train EER: {train_eer.compute().detach().item():.4f}")

        # Reset training metrics
        train_loss.reset()
        train_loss_d.reset()
        train_loss_sce.reset()
        train_auc.reset()
        train_eer.reset()
        if args.personalized:
            train_ckws_auc.reset()
            train_ckws_eer.reset()

        # Validation
        logger.info("Running validation...")
        model.eval()

        # Store metrics for all loaders (Easy and Hard)
        all_loader_metrics = []

        for loader_idx, loader in enumerate(tqdm(eval_dataloader,
                                                   disable=not accelerator.is_local_main_process,dynamic_ncols=True)):
            # Initialize PersonalizedKWSMetrics for this loader
            metrics_calculator = PersonalizedKWSMetrics(fusion_mode=getattr(args, 'fusion_mode', 'multiply'))
            
            # === P_phon 輕量統計累積器 ===
            pphon_positive_means = []
            pphon_positive_mins = []
            pphon_negative_means = []  
            pphon_negative_mins = []

            for batch_idx, batch in enumerate(tqdm(loader,
                                                     desc=f"Val {loader_idx}",
                                                     disable=not accelerator.is_local_main_process,dynamic_ncols=True,leave=False)):
                if batch_idx == 0 and accelerator.is_main_process:
                    print(f"\n>> [DEBUG] Loader {loader_idx}, Batch 0:")
                    print(f">>   batch['z'] unique: {torch.unique(batch['z'])}")
                    print(f">>   batch['z'] mean: {batch['z'].mean():.3f}")
                    if args.personalized and 'category' in batch:
                        from collections import Counter
                        print(f">>   Categories: {Counter(batch['category'])}")
                with torch.no_grad():
                    # Prepare inputs
                    if args.audio_input == "raw":
                        speech_input = batch["x"]
                        speech_len = batch["x_len"]
                    elif args.audio_input == "google_embed":
                        speech_input = batch["gemb"]
                        speech_len = batch["gemb_len"]
                    elif args.audio_input in ("both", "enhanced_gembed"):
                        speech_input = (batch["x"], batch["gemb"])
                        speech_len = (batch["x_len"], batch["gemb_len"])

                    # Forward pass
                    # NOTE: args.mode does NOT affect forward pass - only used for early stopping
                    if args.personalized:
                        # Personalized: always use enrollment_audio (regardless of mode)
                        # All three modes (C-KWS, TB-KWS, TO-KWS) need speaker encoder output
                        output = model(
                            speech=speech_input,
                            text=batch["y"],
                            speech_len=speech_len,
                            text_len=batch["y_len"],
                            enrollment_audio=batch["enrollment_audio"],
                            raw_audio_for_spk=batch.get("x")
                        )
                    else:
                        # Standard PhonMatchNet: no enrollment
                        output = model(
                            speech=speech_input,
                            text=batch["y"],
                            speech_len=speech_len,
                            text_len=batch["y_len"],
                        )

                    # Use score for personalized, P_utt for standard
                    if args.personalized:
                        prob = output['score']  # [B, 1] = P_utt × P_spk
                    else:
                        prob = output['P_utt']  # Standard mode: keyword only

                    # Compute loss for tracking (use LD logit, same as baseline)
                    LD_logit = output['LD']  # Use logit, not probability
                    t_loss, LD_out = loss_object(batch['z'], LD_logit)
                    t_loss /= args.batch_size
                    LD_out /= args.batch_size

                    # Update loss metrics only (proper C-KWS/TB-KWS/TO-KWS metrics use PersonalizedKWSMetrics)
                    test_loss.update(t_loss.item())
                    test_loss_d.update(LD_out.item())

                    # === NEW: Collect personalized metrics ===
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
                    # === FIX: 使用正確的 labels 進行評估 ===
                    # C-KWS: 使用 z_keyword（純 keyword 標籤）
                    # TB-KWS/TO-KWS: 使用 speaker_labels 進行 AND-gated 評估
                    if args.personalized:
                        keyword_labels = batch['z_keyword']  # 純 keyword 標籤
                        speaker_labels = batch.get('speaker_label', torch.zeros_like(keyword_labels))
                        categories = batch.get('category', ['ts-tk'] * len(keyword_labels))
                    else:
                        keyword_labels = batch['z']  # Standard mode 仍用 z
                        speaker_labels = torch.zeros_like(keyword_labels)
                        categories = ['ts-tk'] * len(keyword_labels)

                    # Update PersonalizedKWSMetrics with separate scores
                    metrics_calculator.update(
                        keyword_scores=keyword_scores,
                        speaker_scores=speaker_scores,
                        keyword_labels=keyword_labels,
                        speaker_labels=speaker_labels,
                        categories=categories
                    )
                    
                    # === P_phon 輕量統計收集 ===
                    if 'seq_ce_logit' in output:
                        seq_ce_logit = output['seq_ce_logit']  # [B, T_t]
                        seq_ce_logit_mask = output.get('seq_ce_logit_mask', torch.ones_like(seq_ce_logit))
                        
                        # 轉換為 probability
                        P_phon = torch.sigmoid(seq_ce_logit)  # [B, T_t]
                        
                        # 只計算有效位置（mask=1）
                        P_phon_masked = P_phon * seq_ce_logit_mask
                        valid_counts = seq_ce_logit_mask.sum(dim=-1).clamp(min=1)  # [B]
                        
                        pphon_mean = P_phon_masked.sum(dim=-1) / valid_counts  # [B]
                        
                        # 對於 min，需要處理 mask（設置無效位置為 1.0 以避免影響 min）
                        P_phon_for_min = P_phon.clone()
                        P_phon_for_min[seq_ce_logit_mask == 0] = 1.0
                        pphon_min = P_phon_for_min.min(dim=-1).values  # [B]
                        
                        # 使用 z_keyword 區分正負樣本
                        z_kw = keyword_labels.squeeze(-1) if keyword_labels.dim() > 1 else keyword_labels
                        pos_mask = z_kw == 1
                        neg_mask = z_kw == 0
                        
                        if pos_mask.any():
                            pphon_positive_means.append(pphon_mean[pos_mask].cpu())
                            pphon_positive_mins.append(pphon_min[pos_mask].cpu())
                        if neg_mask.any():
                            pphon_negative_means.append(pphon_mean[neg_mask].cpu())
                            pphon_negative_mins.append(pphon_min[neg_mask].cpu())

            # Compute metrics for all three modes
            all_mode_metrics = metrics_calculator.compute()
            all_loader_metrics.append(all_mode_metrics)

            accelerator.log({
                f"val/{loader_idx}/total": test_loss.compute().detach().item(),
                f"val/{loader_idx}/d": test_loss_d.compute().detach().item(),
            }, step=global_step)

            # Use primary mode EER for scheduler (same as early stopping)
            # NOTE: Scheduler now monitors mode-specific EER instead of C-KWS loss
            if loader_idx == 1 and args.mode in all_mode_metrics:
                val_eer_for_scheduler = all_mode_metrics[args.mode].get('EER', float('inf'))

            # Reset test metrics
            test_loss.reset()
            test_loss_d.reset()
            
            # === P_phon 輕量統計輸出 ===
            if accelerator.is_main_process and (pphon_positive_means or pphon_negative_means):
                loader_name = ['Easy', 'Hard'][loader_idx] if loader_idx < 2 else f'Loader{loader_idx}'
                pphon_stats = compute_pphon_lightweight_stats(
                    pphon_positive_means, pphon_positive_mins,
                    pphon_negative_means, pphon_negative_mins
                )
                print_pphon_lightweight_summary(pphon_stats, loader_name)
                
                # === Save P_phon stats to txt file ===
                pphon_txt_path = os.path.join(args.output_dir, "pphon_stats.txt")
                with open(pphon_txt_path, 'a') as f:
                    f.write(f"Epoch {epoch} - {loader_name}\n")
                    f.write(f"  pos_mean={pphon_stats['positive_mean']:.4f}, "
                            f"neg_mean={pphon_stats['negative_mean']:.4f}, "
                            f"gap_mean={pphon_stats['gap_mean']:.4f}\n")
                    f.write(f"  pos_min={pphon_stats['positive_min']:.4f}, "
                            f"neg_min={pphon_stats['negative_min']:.4f}, "
                            f"gap_min={pphon_stats['gap_min']:.4f}\n")
                
                # Log to tensorboard
                accelerator.log({
                    f"val/{loader_idx}/pphon/positive_mean": pphon_stats['positive_mean'],
                    f"val/{loader_idx}/pphon/negative_mean": pphon_stats['negative_mean'],
                    f"val/{loader_idx}/pphon/gap_mean": pphon_stats['gap_mean'],
                    f"val/{loader_idx}/pphon/positive_min": pphon_stats['positive_min'],
                    f"val/{loader_idx}/pphon/negative_min": pphon_stats['negative_min'],
                    f"val/{loader_idx}/pphon/gap_min": pphon_stats['gap_min'],
                }, step=global_step)

        # === NEW: Print beautiful report for all modes ===
        if accelerator.is_main_process:
            logger.info("\n" + "="*80)
            logger.info("Validation Results (All Modes):")
            logger.info("="*80)

            loader_names = ['Easy', 'Hard']
            for loader_idx, (loader_name, loader_metrics) in enumerate(zip(loader_names, all_loader_metrics)):
                logger.info(f"\n[{loader_name}]")
                for mode in ['C-KWS', 'TB-KWS', 'TO-KWS']:
                    if mode in loader_metrics and 'error' not in loader_metrics[mode]:
                        m = loader_metrics[mode]
                        logger.info(
                            f"  {mode:8s} - EER: {m['EER']*100:6.2f}%  "
                            f"AUC: {m['AUC']*100:6.2f}%  "
                            f"FRR@1%: {m.get('FRR@FAR1%', 0)*100:6.2f}%  "
                            f"FRR@5%: {m.get('FRR@FAR5%', 0)*100:6.2f}%  "
                            f"FRR@10%: {m.get('FRR@FAR10%', 0)*100:6.2f}%"
                        )

                        # Log to tensorboard
                        accelerator.log({
                            f"val/{loader_idx}/{mode}/EER": m['EER'] * 100,
                            f"val/{loader_idx}/{mode}/AUC": m['AUC'] * 100,
                            f"val/{loader_idx}/{mode}/FRR@FAR1%": m.get('FRR@FAR1%', 0) * 100,
                            f"val/{loader_idx}/{mode}/FRR@FAR5%": m.get('FRR@FAR5%', 0) * 100,
                            f"val/{loader_idx}/{mode}/FRR@FAR10%": m.get('FRR@FAR10%', 0) * 100,
                        }, step=global_step)

            logger.info("="*80 + "\n")

        # === 移除 Early Stop，改為 Best Checkpoint Tracking ===
        is_best = False
        if len(all_loader_metrics) > 1 and args.mode in all_loader_metrics[1]:
            primary_eer = all_loader_metrics[1][args.mode].get('EER', float('inf'))

            if primary_eer < best_val_eer:
                improvement = (best_val_eer - primary_eer) * 100
                best_val_eer = primary_eer
                best_metrics = all_loader_metrics
                best_epoch = epoch
                is_best = True

                if accelerator.is_main_process:
                    logger.info(f"✨ New best {args.mode} EER on Hard: {primary_eer*100:.2f}% (improved by {improvement:.2f}%)")
                    
                    # === 更新 best.txt ===
                    best_txt_path = os.path.join(args.output_dir, "best.txt")
                    with open(best_txt_path, 'w') as f:
                        f.write(f"Best Checkpoint Information\n")
                        f.write(f"{'='*60}\n")
                        f.write(f"Epoch: {epoch}\n")
                        f.write(f"Primary Mode: {args.mode}\n")
                        f.write(f"Best EER (Hard): {primary_eer*100:.4f}%\n")
                        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"\n{'='*60}\n")
                        f.write(f"All Metrics at Best Epoch:\n")
                        f.write(f"{'='*60}\n")
                        
                        loader_names = ['Easy', 'Hard']
                        for loader_idx, (loader_name, loader_metrics) in enumerate(zip(loader_names, all_loader_metrics)):
                            f.write(f"\n[{loader_name}]\n")
                            for mode in ['C-KWS', 'TB-KWS', 'TO-KWS']:
                                if mode in loader_metrics and 'error' not in loader_metrics[mode]:
                                    m = loader_metrics[mode]
                                    f.write(
                                        f"  {mode:8s} - EER: {m['EER']*100:6.2f}%  "
                                        f"AUC: {m['AUC']*100:6.2f}%  "
                                        f"FRR@1%: {m.get('FRR@FAR1%', 0)*100:6.2f}%  "
                                        f"FRR@5%: {m.get('FRR@FAR5%', 0)*100:6.2f}%  "
                                        f"FRR@10%: {m.get('FRR@FAR10%', 0)*100:6.2f}%\n"
                                    )
                            # SV 指標
                            if 'SV' in loader_metrics and 'error' not in loader_metrics['SV']:
                                m = loader_metrics['SV']
                                f.write(
                                    f"  {'SV':8s} - EER: {m['EER']*100:6.2f}%  "
                                    f"AUC: {m['AUC']*100:6.2f}%\n"
                                )
                    
                    logger.info(f"📝 Updated best.txt")

        # === 記錄所有 epoch 的驗證結果到 validation_log.txt ===
        if accelerator.is_main_process:
            val_log_path = os.path.join(args.output_dir, "validation_log.txt")
            with open(val_log_path, 'a') as f:  # append mode
                f.write(f"\n{'='*60}\n")
                f.write(f"Epoch {epoch} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*60}\n")
                loader_names = ['Easy', 'Hard']
                for loader_idx, (loader_name, loader_metrics) in enumerate(zip(loader_names, all_loader_metrics)):
                    f.write(f"\n[{loader_name}]\n")
                    for mode in ['C-KWS', 'TB-KWS', 'TO-KWS']:
                        if mode in loader_metrics and 'error' not in loader_metrics[mode]:
                            m = loader_metrics[mode]
                            f.write(
                                f"  {mode:8s} - EER: {m['EER']*100:6.2f}%  "
                                f"AUC: {m['AUC']*100:6.2f}%  "
                                f"FRR@1%: {m.get('FRR@FAR1%', 0)*100:6.2f}%  "
                                f"FRR@5%: {m.get('FRR@FAR5%', 0)*100:6.2f}%  "
                                f"FRR@10%: {m.get('FRR@FAR10%', 0)*100:6.2f}%\n"
                            )
                    # SV 指標
                    if 'SV' in loader_metrics and 'error' not in loader_metrics['SV']:
                        m = loader_metrics['SV']
                        f.write(
                            f"  {'SV':8s} - EER: {m['EER']*100:6.2f}%  "
                            f"AUC: {m['AUC']*100:6.2f}%\n"
                        )

        # === Extended Evaluation: GSC and Qualcomm ===
        if args.eval_gsc_qualcomm and accelerator.is_main_process and extended_loaders is not None:
            logger.info("\n📊 Running extended evaluation (GSC + Qualcomm)...")
            
            extended_metrics_results = {}
            
            # Evaluate GSC
            if extended_loaders.get('google_speech_commands') is not None:
                gsc_metrics = evaluate_extended_dataset(
                    model, extended_loaders['google_speech_commands'], 
                    'GSC', args, accelerator
                )
                extended_metrics_results['GSC'] = gsc_metrics
                
                if gsc_metrics is not None:
                    # Log to TensorBoard
                    for mode in ['C-KWS', 'TB-KWS', 'TO-KWS']:
                        if mode in gsc_metrics and 'EER' in gsc_metrics[mode]:
                            m = gsc_metrics[mode]
                            accelerator.log({
                                f"val/gsc/{mode}/EER": m['EER'] * 100,
                                f"val/gsc/{mode}/AUC": m.get('AUC', 0) * 100,
                                f"val/gsc/{mode}/FRR@FAR1%": m.get('FRR@FAR1%', 0) * 100,
                                f"val/gsc/{mode}/FRR@FAR5%": m.get('FRR@FAR5%', 0) * 100,
                                f"val/gsc/{mode}/FRR@FAR10%": m.get('FRR@FAR10%', 0) * 100,
                            }, step=global_step)
                    
                    # Print results
                    logger.info("\n[GSC]")
                    for mode in ['C-KWS', 'TB-KWS', 'TO-KWS']:
                        if mode in gsc_metrics and 'EER' in gsc_metrics[mode]:
                            m = gsc_metrics[mode]
                            logger.info(
                                f"  {mode:8s} - EER: {m['EER']*100:6.2f}%  "
                                f"AUC: {m.get('AUC', 0)*100:6.2f}%  "
                                f"FRR@1%: {m.get('FRR@FAR1%', 0)*100:6.2f}%  "
                                f"FRR@5%: {m.get('FRR@FAR5%', 0)*100:6.2f}%  "
                                f"FRR@10%: {m.get('FRR@FAR10%', 0)*100:6.2f}%"
                            )
            
            # Evaluate Qualcomm
            if extended_loaders.get('qualcomm') is not None:
                qualcomm_metrics = evaluate_extended_dataset(
                    model, extended_loaders['qualcomm'], 
                    'Qualcomm', args, accelerator
                )
                extended_metrics_results['Qualcomm'] = qualcomm_metrics
                
                if qualcomm_metrics is not None:
                    # Log to TensorBoard
                    for mode in ['C-KWS', 'TB-KWS', 'TO-KWS']:
                        if mode in qualcomm_metrics and 'EER' in qualcomm_metrics[mode]:
                            m = qualcomm_metrics[mode]
                            accelerator.log({
                                f"val/qualcomm/{mode}/EER": m['EER'] * 100,
                                f"val/qualcomm/{mode}/AUC": m.get('AUC', 0) * 100,
                                f"val/qualcomm/{mode}/FRR@FAR1%": m.get('FRR@FAR1%', 0) * 100,
                                f"val/qualcomm/{mode}/FRR@FAR5%": m.get('FRR@FAR5%', 0) * 100,
                                f"val/qualcomm/{mode}/FRR@FAR10%": m.get('FRR@FAR10%', 0) * 100,
                            }, step=global_step)
                    
                    # Print results
                    logger.info("\n[Qualcomm]")
                    for mode in ['C-KWS', 'TB-KWS', 'TO-KWS']:
                        if mode in qualcomm_metrics and 'EER' in qualcomm_metrics[mode]:
                            m = qualcomm_metrics[mode]
                            logger.info(
                                f"  {mode:8s} - EER: {m['EER']*100:6.2f}%  "
                                f"AUC: {m.get('AUC', 0)*100:6.2f}%  "
                                f"FRR@1%: {m.get('FRR@FAR1%', 0)*100:6.2f}%  "
                                f"FRR@5%: {m.get('FRR@FAR5%', 0)*100:6.2f}%  "
                                f"FRR@10%: {m.get('FRR@FAR10%', 0)*100:6.2f}%"
                            )
            
            # Write extended results to validation_log.txt
            val_log_path = os.path.join(args.output_dir, "validation_log.txt")
            with open(val_log_path, 'a') as f:
                for dataset_name, metrics in extended_metrics_results.items():
                    if metrics is not None:
                        f.write(f"\n[{dataset_name}]\n")
                        for mode in ['C-KWS', 'TB-KWS', 'TO-KWS']:
                            if mode in metrics and 'EER' in metrics[mode]:
                                m = metrics[mode]
                                f.write(
                                    f"  {mode:8s} - EER: {m['EER']*100:6.2f}%  "
                                    f"AUC: {m.get('AUC', 0)*100:6.2f}%  "
                                    f"FRR@1%: {m.get('FRR@FAR1%', 0)*100:6.2f}%  "
                                    f"FRR@5%: {m.get('FRR@FAR5%', 0)*100:6.2f}%  "
                                    f"FRR@10%: {m.get('FRR@FAR10%', 0)*100:6.2f}%\n"
                                )
                        # SV 指標
                        if 'SV' in metrics and 'EER' in metrics['SV']:
                            m = metrics['SV']
                            f.write(
                                f"  {'SV':8s} - EER: {m['EER']*100:6.2f}%  "
                                f"AUC: {m.get('AUC', 0)*100:6.2f}%\n"
                            )
            
            # Append GSC/Qualcomm results to best.txt if this is best epoch
            if is_best and extended_metrics_results:
                best_txt_path = os.path.join(args.output_dir, "best.txt")
                with open(best_txt_path, 'a') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"Extended Evaluation (GSC + Qualcomm):\n")
                    f.write(f"{'='*60}\n")
                    for dataset_name, metrics in extended_metrics_results.items():
                        if metrics is not None:
                            f.write(f"\n[{dataset_name}]\n")
                            for mode in ['C-KWS', 'TB-KWS', 'TO-KWS']:
                                if mode in metrics and 'EER' in metrics[mode]:
                                    m = metrics[mode]
                                    f.write(
                                        f"  {mode:8s} - EER: {m['EER']*100:6.2f}%  "
                                        f"AUC: {m.get('AUC', 0)*100:6.2f}%  "
                                        f"FRR@1%: {m.get('FRR@FAR1%', 0)*100:6.2f}%  "
                                        f"FRR@5%: {m.get('FRR@FAR5%', 0)*100:6.2f}%  "
                                        f"FRR@10%: {m.get('FRR@FAR10%', 0)*100:6.2f}%\n"
                                    )
                            # SV 指標
                            if 'SV' in metrics and 'EER' in metrics['SV']:
                                m = metrics['SV']
                                f.write(
                                    f"  {'SV':8s} - EER: {m['EER']*100:6.2f}%  "
                                    f"AUC: {m.get('AUC', 0)*100:6.2f}%\n"
                                )
            
            logger.info("=" * 80)

        # Update learning rate
        if scheduler is not None:
            if args.scheduler_type == 'plateau':
                # ReduceLROnPlateau needs val metric
                if 'val_eer_for_scheduler' in locals():
                    scheduler.step(val_eer_for_scheduler)
            elif args.scheduler_type == 'cosine':
                # CosineAnnealingLR: step each epoch
                scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"📉 Current LR: {current_lr:.2e}")

        # Free memory
        torch.cuda.empty_cache()
        gc.collect()

        # === Save checkpoint ===
        save_path = os.path.join(args.output_dir, f"checkpoint_{epoch}.pth")
        logger.info(f"💾 Saving checkpoint: {save_path}")

        checkpoint_data = {
            'model': accelerator.unwrap_model(model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': global_step,
            'args': vars(args),
            'best_val_eer': best_val_eer,
            'best_metrics': best_metrics,
            'best_epoch': best_epoch if 'best_epoch' in locals() else epoch,
        }
        
        torch.save(checkpoint_data, save_path)

        # === 如果是最佳，另存一份 best_checkpoint.pth ===
        if is_best:
            best_ckpt_path = os.path.join(args.output_dir, "best_checkpoint.pth")
            torch.save(checkpoint_data, best_ckpt_path)
            logger.info(f"🏆 Saved best checkpoint: {best_ckpt_path}")

        # === Disabled: 刪除舊的 checkpoint（保留所有）===
        # all_ckpts = sorted(
        #     [p for p in Path(args.output_dir).glob("checkpoint_*.pth") if 'best' not in p.name],
        #     key=lambda p: int(p.stem.split('_')[1])
        # )
        # if len(all_ckpts) > 3:
        #     for old_ckpt in all_ckpts[:-3]:
        #         os.remove(old_ckpt)
        #         logger.info(f"🗑️ Removed old checkpoint: {old_ckpt}")

    accelerator.wait_for_everyone()
    logger.info("🎉 Training complete!")
    return


if __name__ == "__main__":
    main()