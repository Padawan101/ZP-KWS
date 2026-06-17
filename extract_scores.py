"""
Score Extraction for DET Curve Generation

Extract raw scores from three model types for DET curve comparison:
  - phonmatchnet: BaseUKWS from /padawan/PhonMatchNet — score = P_utt
  - pkmtl: PhonMatchNetPKMTL from /padawan/PhonMatchNet_PK-MTL — score = α·P_utt + (1-α)·ψ_s (SCM)
  - p_ukws: P_UKWS from /padawan/PKWS — score = P_utt × P_spk (multiplicative)

All systems use the SAME dataloader (PKWS PersonalizedLibriPhraseDataset, Hard split)
to ensure identical sample ordering and speaker pairing for paired statistical tests.

Output .npz format:
  score:           [N] final TO-KWS score (system-specific fusion)
  P_utt:           [N] keyword probability
  keyword_labels:  [N] binary
  speaker_labels:  [N] binary
  categories:      [N] str

Usage:
    python extract_scores.py \
        --checkpoint results/phonmatchnet/checkpoint_27.pth \
        --model_type phonmatchnet \
        --output results/scores/phonmatchnet_hard.npz
"""

import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm


_CODEBASE_DIRS = ['/padawan/PhonMatchNet', '/padawan/PhonMatchNet_PK-MTL', '/padawan/PKWS']


def _switch_codebase(codebase_dir):
    """
    Switch Python import context to a different codebase.

    Three codebases share package names (`model/`, `dataset/`, `utils`, etc.)
    AND use bare imports (`import encoder`, `from utils import ...`).
    This function:
    1. Removes ALL cached modules whose __file__ is inside any of the 3 codebases
    2. Removes other codebase dirs from sys.path
    3. Inserts the target codebase (and its model/ subdir) at the front
    """
    # Step 1: purge all modules loaded from any codebase
    for key in list(sys.modules.keys()):
        mod = sys.modules[key]
        mod_file = getattr(mod, '__file__', None) or ''
        if any(mod_file.startswith(d) for d in _CODEBASE_DIRS):
            del sys.modules[key]

    # Step 2: remove other codebase dirs (and their model/ subdirs) from sys.path
    dirs_to_remove = []
    for d in _CODEBASE_DIRS:
        dirs_to_remove.extend([d, os.path.join(d, 'model')])
    sys.path = [p for p in sys.path if p not in dirs_to_remove]

    # Step 3: insert target codebase + model/ subdir at front
    # model/ subdir is needed for PK-MTL bare imports like `import encoder`
    model_dir = os.path.join(codebase_dir, 'model')
    if os.path.isdir(model_dir):
        sys.path.insert(0, model_dir)
    sys.path.insert(0, codebase_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract scores for DET curves")
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--model_type', required=True, type=str,
                        choices=['phonmatchnet', 'pkmtl', 'p_ukws'])
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--libriphrase_pkl', type=str, default='/padawan/test_500h.pkl')
    parser.add_argument('--train_pkl', type=str, default='/padawan/train_combined.pkl')
    return parser.parse_args()


# ============================================================
# Unified Dataloader (PKWS — all systems use this)
# ============================================================

def create_unified_hard_loader(args):
    """
    Create LibriPhrase-Hard loader using PKWS PersonalizedLibriPhraseDataset.
    All systems share this loader for sample alignment.
    """
    _switch_codebase('/padawan/PKWS')

    pkws_dir = '/padawan/PKWS'
    if pkws_dir not in sys.path:
        sys.path.insert(0, pkws_dir)

    from dataset.personalized_libriphrase import PersonalizedLibriPhraseDataset, SubsetDataset
    from dataset import KWSDataLoader

    gemb_dir = '/padawan/google_speech_embedding/DB/'

    base_val_dataset = PersonalizedLibriPhraseDataset(
        batch_size=args.batch_size,
        gemb_dir=gemb_dir,
        features='g2p_embed',
        train=False,
        types='both',
        shuffle=False,
        pkl=args.libriphrase_pkl,
        frame_length=400,
        hop_length=160,
        personalized=True,
        speaker_ratio=0.5,
    )

    # Extract hard subset
    hard_indices = []
    for i, sample_type in enumerate(base_val_dataset.type_list):
        if 'hard' in str(sample_type):
            hard_indices.append(i)

    val_hard_dataset = SubsetDataset(base_val_dataset, hard_indices)
    print(f">> Hard subset: {len(hard_indices)} samples")

    val_hard_loader = KWSDataLoader(
        val_hard_dataset,
        args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    vocab = base_val_dataset.nPhoneme

    # Keep dataset reference for PK-MTL gemb loading
    return val_hard_loader, vocab, base_val_dataset


# ============================================================
# Model Loaders
# ============================================================

def load_phonmatchnet(ckpt_path, vocab, device):
    """Load PhonMatchNet (BaseUKWS) from /padawan/PhonMatchNet"""
    _switch_codebase('/padawan/PhonMatchNet')
    from model.ukws import BaseUKWS

    kwargs = {
        'vocab': vocab,
        'text_input': 'g2p_embed',
        'audio_input': 'both',
        'stack_extractor': False,
        'frame_length': 400,
        'hop_length': 160,
        'num_mel': 40,
        'sample_rate': 16000,
        'log_mel': False,
    }

    model = BaseUKWS(**kwargs)
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', 'N/A')
    print(f">> Loaded PhonMatchNet (BaseUKWS): epoch={epoch}")
    return model


def load_pkmtl(ckpt_path, vocab, num_speakers, device):
    """Load PK-MTL (PhonMatchNetPKMTL) from /padawan/PhonMatchNet_PK-MTL"""
    _switch_codebase('/padawan/PhonMatchNet_PK-MTL')
    from model.pkmtl import PhonMatchNetPKMTL

    kwargs = {
        'vocab': vocab,
        'text_input': 'g2p_embed',
        'audio_input': 'both',
        'stack_extractor': False,
        'frame_length': 400,
        'hop_length': 160,
        'num_mel': 40,
        'sample_rate': 16000,
        'log_mel': False,
    }

    model = PhonMatchNetPKMTL(
        num_speakers=num_speakers,
        sv_input='conv_only',
        sv_hidden_dim=256,
        sv_output_dim=128,
        sv_pooling='attention',
        scm_alpha=0.5,
        mode='TO-KWS',
        **kwargs,
    )

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', 'N/A')
    alpha = model.get_alpha()
    print(f">> Loaded PK-MTL: epoch={epoch}, SCM α={alpha:.3f}, num_speakers={num_speakers}")
    return model


def load_p_ukws(ckpt_path, device):
    """Load P_UKWS from /padawan/PKWS (uses saved args from checkpoint)"""
    pkws_dir = '/padawan/PKWS'
    if pkws_dir not in sys.path:
        sys.path.insert(0, pkws_dir)

    _switch_codebase('/padawan/PKWS')
    from model.p_ukws import P_UKWS

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    saved_args = checkpoint['args']

    kwargs = {
        'vocab': saved_args.get('vocab', 42),
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
        'disable_film': saved_args.get('disable_film', False),
        'disable_sv_branch': saved_args.get('disable_sv_branch', False),
        'freeze_speaker_encoder': saved_args.get('freeze_speaker_encoder', True),
        'finetuned_speaker_encoder_path': saved_args.get('finetuned_speaker_encoder', None),
        'film_target': saved_args.get('film_target', 'fused'),
        'film_gate_type': saved_args.get('film_gate_type', 'pspk'),
        'enable_aux_ce': saved_args.get('enable_aux_ce', False),
        'n_phonemes': saved_args.get('n_phonemes', 42),
        'gemb_drop_rate': saved_args.get('gemb_drop_rate', 0.0),
        'gemb_curriculum': saved_args.get('gemb_curriculum', False),
        'gemb_warmup_epochs': saved_args.get('gemb_warmup_epochs', 5),
        'gemb_ramp_epochs': saved_args.get('gemb_ramp_epochs', 10),
        'calibration_mode': saved_args.get('calibration_mode', 'full'),
        'fusion_mode': saved_args.get('fusion_mode', 'multiply'),
        'disable_hybrid_encoder': saved_args.get('disable_hybrid_encoder', False),
        'disable_ldn_norm': saved_args.get('disable_ldn_norm', False),
        'gru_layers': saved_args.get('gru_layers', 2),
        'bidirectional': saved_args.get('bidirectional', False),
        'stream_fusion': saved_args.get('stream_fusion', 'add'),
    }

    model = P_UKWS(**kwargs)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', 'N/A')
    print(f">> Loaded P_UKWS: epoch={epoch}, "
          f"disable_film={kwargs['disable_film']}, disable_sv={kwargs['disable_sv_branch']}")
    return model


def get_pkmtl_num_speakers(train_pkl):
    """Get num_speakers from PK-MTL training dataset"""
    _switch_codebase('/padawan/PhonMatchNet_PK-MTL')
    from dataset.pkmtl_libriphrase import PKMTLLibriPhraseDataset

    train_ds = PKMTLLibriPhraseDataset(
        batch_size=1,
        gemb_dir='/padawan/google_speech_embedding/DB/',
        features='g2p_embed',
        train=True,
        types='both',
        shuffle=False,
        pkl=train_pkl,
        frame_length=400,
        hop_length=160,
        personalized=False,
    )
    n = train_ds.num_speakers
    print(f">> PK-MTL num_speakers: {n}")
    del train_ds
    return n





# ============================================================
# Forward Pass Functions
# ============================================================

def _move_batch(batch, device):
    """Move tensor fields to device"""
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)


def run_forward_phonmatchnet(model, loader, device):
    """
    PhonMatchNet: BaseUKWS, tuple output, no speaker branch.
    score = P_utt (no fusion)
    """
    all_score, all_p_utt = [], []
    all_kw, all_spk, all_cats = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="PhonMatchNet forward"):
            _move_batch(batch, device)

            speech = (batch["x"], batch["gemb"])
            speech_len = (batch["x_len"], batch["gemb_len"])

            prob, *_ = model(speech, batch["y"], speech_len, batch["y_len"])

            P_utt = prob.squeeze(-1).cpu().numpy()
            score = P_utt.copy()  # No speaker branch → score = P_utt

            all_score.append(score)
            all_p_utt.append(P_utt)
            all_kw.append(batch['z_keyword'].squeeze(-1).cpu().numpy())
            all_spk.append(batch['speaker_label'].squeeze(-1).cpu().numpy())
            all_cats.extend(batch['category'])

    return _concat(all_score, all_p_utt, all_kw, all_spk, all_cats)


def run_forward_pkmtl(model, loader, device, base_dataset):
    """
    PK-MTL: PhonMatchNetPKMTL, dict output.
    score = α × P_utt + (1-α) × ψ_s (SCM fusion)

    The PKWS dataloader only provides raw enrollment_audio (no gemb).
    PK-MTL in 'both' mode needs (audio, gemb) tuple for AE forward.

    We pass a zero-padded fake gemb through AE with return_conv_feat=True.
    This is SAFE because sv_input='conv_only' means the SV subnet only
    receives conv_feat, which is cloned at EfficientAudioEncoder L153
    BEFORE gemb fusion happens at L157-172. Fake gemb does not affect
    conv_feat or z_enroll.
    """
    all_score, all_p_utt = [], []
    all_kw, all_spk, all_cats = [], [], []

    alpha = model.get_alpha()
    print(f">> PK-MTL SCM alpha: {alpha:.4f}")
    print(f">> PK-MTL fusion: score = {alpha:.3f} × P_utt + {1-alpha:.3f} × ψ_s")

    with torch.no_grad():
        for batch in tqdm(loader, desc="PK-MTL forward"):
            _move_batch(batch, device)

            speech = (batch["x"], batch["gemb"])
            speech_len = (batch["x_len"], batch["gemb_len"])

            # 2-step approach:
            #   1. Forward WITHOUT enrollment → get P_utt and z_spk
            #   2. Process enrollment through SPEC → AE(return_conv_feat=True) → SV
            #      conv_feat is independent of gemb (see EfficientAudioEncoder L153)
            
            # Step 1: forward without enrollment to get P_utt, z_spk, etc.
            output_no_enroll = model(
                speech=speech,
                text=batch["y"],
                speech_len=speech_len,
                text_len=batch["y_len"],
                enrollment_audio=None,
                return_all=True,
            )

            P_utt = output_no_enroll['P_utt'].squeeze(-1)  # [B]
            z_spk = output_no_enroll['z_spk']               # [B, D]

            # Step 2: get enrollment speaker embedding
            # Process enrollment audio through SPEC → AudioEncoder → SV subnet
            enroll_raw = batch["enrollment_audio"]           # [B, T]
            enroll_spec, enroll_s_mask = model.SPEC(enroll_raw, False)

            # AE requires (spec, gemb) tuple for 'both' mode.
            # Fake gemb is safe: with return_conv_feat=True, conv_feat is cloned
            # BEFORE gemb fusion (EfficientAudioEncoder L153 vs L157-172).
            target_len = enroll_spec.shape[1] // 8
            if target_len < 1:
                target_len = 1
            fake_gemb = torch.zeros(enroll_raw.shape[0], target_len, 96, device=device)
            fake_gemb_len = torch.ones(enroll_raw.shape[0], dtype=torch.int32, device=device) * target_len
            fake_gemb_mask = torch.ones(enroll_raw.shape[0], target_len, dtype=torch.bool, device=device)

            # Run through audio encoder
            use_conv_feat = (model.sv_input == 'conv_only')
            if use_conv_feat:
                enroll_fused, enroll_conv, _, enroll_mask = model.AE(
                    (enroll_spec, fake_gemb), (enroll_s_mask, fake_gemb_mask),
                    False, return_conv_feat=True
                )
            else:
                enroll_fused, _, enroll_mask = model.AE(
                    (enroll_spec, fake_gemb), (enroll_s_mask, fake_gemb_mask),
                    False, return_conv_feat=False
                )
                enroll_conv = None

            enroll_sv_input = enroll_conv if (use_conv_feat and enroll_conv is not None) else enroll_fused
            z_enroll = model.sv_subnet(enroll_sv_input, mask=enroll_mask)

            # Compute speaker score (cosine similarity)
            _switch_codebase('/padawan/PhonMatchNet_PK-MTL')
            from model.sv_subnet import compute_speaker_score
            psi_s = compute_speaker_score(z_spk, z_enroll)  # [B]

            # SCM fusion: score = α × P_utt + (1-α) × ψ_s
            psi_k = P_utt  # [B]
            score = model.scm(psi_k, psi_s)  # Uses learned alpha

            all_score.append(score.cpu().numpy())
            all_p_utt.append(P_utt.cpu().numpy())
            all_kw.append(batch['z_keyword'].squeeze(-1).cpu().numpy())
            all_spk.append(batch['speaker_label'].squeeze(-1).cpu().numpy())
            all_cats.extend(batch['category'])

    return _concat(all_score, all_p_utt, all_kw, all_spk, all_cats)


def run_forward_p_ukws(model, loader, device):
    """
    P_UKWS: dict output.
    score = P_utt × P_spk (multiplicative fusion)
    """
    all_score, all_p_utt = [], []
    all_kw, all_spk, all_cats = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="P_UKWS forward"):
            _move_batch(batch, device)

            speech = (batch["x"], batch["gemb"])
            speech_len = (batch["x_len"], batch["gemb_len"])

            output = model(
                speech=speech,
                text=batch["y"],
                speech_len=speech_len,
                text_len=batch["y_len"],
                enrollment_audio=batch["enrollment_audio"],
                raw_audio_for_spk=batch.get("x"),
            )

            P_utt = output['P_utt'].squeeze(-1)  # [B]
            P_spk = output.get('P_spk', torch.ones_like(P_utt))
            P_spk = P_spk.squeeze(-1)  # [B]

            score = (P_utt * P_spk).cpu().numpy()  # Multiplicative fusion

            all_score.append(score)
            all_p_utt.append(P_utt.cpu().numpy())
            all_kw.append(batch['z_keyword'].squeeze(-1).cpu().numpy())
            all_spk.append(batch['speaker_label'].squeeze(-1).cpu().numpy())
            all_cats.extend(batch['category'])

    return _concat(all_score, all_p_utt, all_kw, all_spk, all_cats)


def _concat(all_score, all_p_utt, all_kw, all_spk, all_cats):
    return {
        'score': np.concatenate(all_score).astype(np.float32),
        'P_utt': np.concatenate(all_p_utt).astype(np.float32),
        'keyword_labels': np.concatenate(all_kw).astype(int),
        'speaker_labels': np.concatenate(all_spk).astype(int),
        'categories': np.array(all_cats),
    }


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print(f"Score Extraction: {args.model_type}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output:     {args.output}")
    print(f"  Device:     {device}")
    print("=" * 60)

    # Unified dataloader (all systems use same samples)
    loader, vocab, base_dataset = create_unified_hard_loader(args)

    if args.model_type == 'phonmatchnet':
        model = load_phonmatchnet(args.checkpoint, vocab, device)
        results = run_forward_phonmatchnet(model, loader, device)

    elif args.model_type == 'pkmtl':
        num_speakers = get_pkmtl_num_speakers(args.train_pkl)
        model = load_pkmtl(args.checkpoint, vocab, num_speakers, device)
        results = run_forward_pkmtl(model, loader, device, base_dataset)

    elif args.model_type == 'p_ukws':
        model = load_p_ukws(args.checkpoint, device)
        results = run_forward_p_ukws(model, loader, device)

    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    np.savez(
        args.output,
        score=results['score'],
        P_utt=results['P_utt'],
        keyword_labels=results['keyword_labels'],
        speaker_labels=results['speaker_labels'],
        categories=results['categories'],
    )

    # Sanity check
    N = len(results['score'])
    cats = results['categories']
    print(f"\n{'='*60}")
    print(f"Saved {N} samples to {args.output}")
    print(f"  score range:  [{results['score'].min():.4f}, {results['score'].max():.4f}]")
    print(f"  P_utt range:  [{results['P_utt'].min():.4f}, {results['P_utt'].max():.4f}]")
    print(f"  Category distribution:")
    for cat in ['ts-tk', 'nts-tk', 'ts-ntk', 'nts-ntk']:
        count = np.sum(cats == cat)
        print(f"    {cat}: {count} ({count/N*100:.1f}%)")
    print(f"  keyword_labels: {results['keyword_labels'].sum()}/{N} positive")
    print(f"  speaker_labels: {results['speaker_labels'].sum()}/{N} positive")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
