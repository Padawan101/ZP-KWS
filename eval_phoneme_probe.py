"""
Phoneme Linear Probe - 測量 Audio Encoder 特徵的音素辨識能力

輸出：
    LDN phoneme accuracy:     XX.XX%  (trainable conv features only)
    emb_s phoneme accuracy:   XX.XX%  (LDN + gembed fused)
    
    若 emb_s acc >> LDN acc → gembed 主導音素辨識，Aux CE 需要更強的介入
    若 emb_s acc ≈ LDN acc → gembed 不擅長音素，Aux CE 有很大的改善空間
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Project imports
from model import p_ukws
from dataset.personalized_libriphrase import PersonalizedLibriPhraseDataset
from dataset import KWSDataLoader


def align_labels(frame_labels, target_len):
    """
    對齊 frame labels 到 encoder 輸出長度。
    
    使用截斷/補零（與訓練時 AuxCELoss 中的 align_labels_to_encoder 一致）。
    不能用 F.interpolate，因為 frame_labels 長度 < target_len 時
    是 padding 造成的（labels 只覆蓋實際語音部分），不是解析度差異。
    Interpolate 會把短 labels 拉伸到整段長度，導致完全錯位。
    """
    B, T = frame_labels.shape
    if T == target_len:
        return frame_labels
    elif T > target_len:
        return frame_labels[:, :target_len]
    else:
        return F.pad(frame_labels, (0, target_len - T), value=0)


class FeatureCapture:
    """
    用 forward hook 擷取 AudioEncoder 的中間輸出。
    
    AudioEncoder forward 回傳: (emb_s, LDN, mask) 或 (emb_s, LDN, gembed_proc, mask)
    我們擷取 emb_s (index 0) 和 LDN (index 1)。
    """
    def __init__(self):
        self.emb_s = None
        self.ldn = None
        self.hook = None
    
    def hook_fn(self, module, input, output):
        """AudioEncoder 的 output 是 tuple: (emb_s, LDN, [gembed_proc,] mask)"""
        if isinstance(output, tuple) and len(output) >= 3:
            self.emb_s = output[0].detach()  # [B, T, 128]
            self.ldn = output[1].detach()     # [B, T, 128]
    
    def register(self, model):
        """註冊 hook 到 model.AE"""
        raw_model = model.module if hasattr(model, 'module') else model
        self.hook = raw_model.AE.register_forward_hook(self.hook_fn)
        return self
    
    def remove(self):
        if self.hook is not None:
            self.hook.remove()


class LinearProbe(nn.Module):
    """簡單的 Linear probe for phoneme classification"""
    def __init__(self, feature_dim=128, n_phonemes=42):
        super().__init__()
        self.linear = nn.Linear(feature_dim, n_phonemes)
    
    def forward(self, x):
        return self.linear(x)  # [B, T, n_phonemes]


def evaluate_probe(probe, features_list, labels_list, device):
    """在收集好的特徵上評估 probe accuracy"""
    probe.eval()
    total_correct = 0
    total_valid = 0
    total_active_correct = 0
    total_active = 0
    
    with torch.no_grad():
        for features, labels in zip(features_list, labels_list):
            logits = probe(features.to(device))
            logits_flat = logits.reshape(-1, logits.shape[-1])
            labels_flat = labels.reshape(-1).to(device)
            
            preds = logits_flat.argmax(dim=-1)
            valid = labels_flat != 0  # ignore PAD
            if valid.any():
                total_correct += (preds[valid] == labels_flat[valid]).sum().item()
                total_valid += valid.sum().item()
            
            # Active accuracy (non-SIL/SPN)
            sil_mask = (labels_flat == 1) | (labels_flat == 2)
            active = valid & (~sil_mask)
            if active.any():
                total_active_correct += (preds[active] == labels_flat[active]).sum().item()
                total_active += active.sum().item()
    
    acc = total_correct / max(total_valid, 1)
    acc_active = total_active_correct / max(total_active, 1)
    return acc, acc_active


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # ===== 1. 載入 Checkpoint =====
    print(f"\n{'='*60}")
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # 從 checkpoint 取得 model config
    train_args = checkpoint.get('args', None)
    
    if train_args is None:
        print("⚠️ No training args in checkpoint, using defaults")
        train_args = {}
    
    # train_args 是 dict (因為存的是 vars(args))
    ta = train_args if isinstance(train_args, dict) else vars(train_args)
    
    # ===== 2. 建立 Dataset =====
    print(f"\nLoading dataset...")
    
    audio_input = ta.get('audio_input', 'both')
    gemb_dir = None if audio_input == 'raw' else '/padawan/google_speech_embedding/DB/'
    
    train_dataset = PersonalizedLibriPhraseDataset(
        batch_size=args.batch_size,
        gemb_dir=gemb_dir,
        features=ta.get('text_input', 'g2p_embed'),
        train=True,
        types='both',
        shuffle=True,
        pkl=ta.get('train_pkl', None),
        frame_length=ta.get('frame_length', 400),
        hop_length=ta.get('hop_length', 160),
        personalized=False,  # Probe 不需要 personalized
        frame_labels_path=args.frame_labels_path,
    )
    
    vocab = train_dataset.nPhoneme
    print(f"  vocab (G2P phonemes): {vocab}")
    
    dataloader = KWSDataLoader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=True,
    )
    
    # ===== 3. 建立並載入模型 =====
    print(f"\nBuilding model...")
    model_kwargs = {
        'vocab': vocab,
        'text_input': ta.get('text_input', 'g2p_embed'),
        'audio_input': audio_input,
        'stack_extractor': ta.get('stack_extractor', True),
        'frame_length': ta.get('frame_length', 400),
        'hop_length': ta.get('hop_length', 160),
        'num_mel': 40,
        'sample_rate': ta.get('sample_rate', 16000),
        'log_mel': ta.get('log_mel', True),
        'mode': ta.get('mode', 'C-KWS'),
        'speaker_encoder_path': ta.get('speaker_encoder_path', 'model/speaker/efficient_tdnn'),
        'disable_film': ta.get('disable_film', False),
        'disable_sv_branch': ta.get('disable_sv_branch', False),
        'freeze_speaker_encoder': True,
        'finetuned_speaker_encoder_path': ta.get('finetuned_speaker_encoder', None),
        'film_target': ta.get('film_target', 'fused'),
        'film_gate_type': ta.get('film_gate_type', 'pspk'),
        'enable_aux_ce': False,  # Probe 不需要模型自帶的 aux head
        'n_phonemes': 42,
        'gemb_drop_rate': 0.0,  # Probe 不需要 dropout
        'gemb_curriculum': False,  # Probe 不需要 curriculum
        'bidirectional': ta.get('bidirectional', False),
        'gru_layers': ta.get('gru_layers', 2),
        'disable_hybrid_encoder': ta.get('disable_hybrid_encoder', False),
        'disable_ldn_norm': ta.get('disable_ldn_norm', False),
    }
    
    model = p_ukws.P_UKWS(**model_kwargs)
    
    # 載入 weights（忽略 aux_head 等不存在的 key）
    model_state = checkpoint.get('model', checkpoint.get('model_state_dict', checkpoint))
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print(f"  Loaded model weights")
    if missing:
        print(f"  Missing keys: {len(missing)} (expected if aux_head/ldn_norm not in checkpoint)")
        for k in missing[:5]:
            print(f"    - {k}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"    - {k}")
    
    model = model.to(device)
    model.eval()
    
    # Freeze 整個模型
    for param in model.parameters():
        param.requires_grad = False
    
    # ===== 4. 註冊 Feature Capture Hook =====
    capture = FeatureCapture().register(model)
    
    # ===== 5. 收集特徵 =====
    print(f"\nCollecting features from {args.n_collect_batches} batches...")
    
    ldn_features_list = []
    embs_features_list = []
    labels_list = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, total=args.n_collect_batches, desc="Collecting")):
            if i >= args.n_collect_batches:
                break
            
            # 準備 input
            if audio_input == 'both':
                speech_input = (batch['x'].to(device), batch['gemb'].to(device))
                speech_len = (batch['x_len'].to(device), batch['gemb_len'].to(device))
            elif audio_input == 'raw':
                speech_input = batch['x'].to(device)
                speech_len = batch['x_len'].to(device)
            else:
                speech_input = batch['gemb'].to(device)
                speech_len = batch['gemb_len'].to(device)
            
            text_input = batch['y'].to(device)
            text_len = batch['y_len'].to(device)
            frame_labels = batch.get('frame_labels')
            
            if frame_labels is None:
                continue
            
            # Forward pass（觸發 hook）
            _ = model(speech_input, text_input, speech_len, text_len)
            
            # 從 hook 取得特徵
            if capture.ldn is not None and capture.emb_s is not None:
                T_a = capture.ldn.shape[1]
                aligned_labels = align_labels(frame_labels, T_a)
                
                ldn_features_list.append(capture.ldn.cpu())
                embs_features_list.append(capture.emb_s.cpu())
                labels_list.append(aligned_labels.cpu())
    
    capture.remove()
    
    if not labels_list:
        print("❌ No features collected! Check batch structure and hook registration.")
        return
    
    print(f"  Collected {len(labels_list)} batches")
    print(f"  LDN shape: {ldn_features_list[0].shape}")
    print(f"  emb_s shape: {embs_features_list[0].shape}")
    
    # 計算特徵 magnitude
    ldn_mag = torch.cat([f.abs().mean(dim=(1,2)) for f in ldn_features_list]).mean().item()
    embs_mag = torch.cat([f.abs().mean(dim=(1,2)) for f in embs_features_list]).mean().item()
    print(f"\n📏 Feature Magnitude:")
    print(f"  LDN:   {ldn_mag:.4f}")
    print(f"  emb_s: {embs_mag:.4f}")
    print(f"  ratio: {ldn_mag / (embs_mag + 1e-8):.4f}")
    
    # ===== 6. 訓練 Linear Probes =====
    print(f"\n{'='*60}")
    print(f"Training Linear Probes ({args.probe_epochs} epochs)")
    print(f"{'='*60}")
    
    # 分割 train/val（前 80% 訓練，後 20% 驗證）
    n_train = int(len(labels_list) * 0.8)
    
    train_ldn = ldn_features_list[:n_train]
    train_embs = embs_features_list[:n_train]
    train_labels = labels_list[:n_train]
    
    val_ldn = ldn_features_list[n_train:]
    val_embs = embs_features_list[n_train:]
    val_labels = labels_list[n_train:]
    
    results = {}
    
    for probe_name, train_feats, val_feats in [
        ("LDN (conv only)", train_ldn, val_ldn),
        ("emb_s (LDN + gembed)", train_embs, val_embs),
    ]:
        print(f"\n--- Probe: {probe_name} ---")
        probe = LinearProbe(feature_dim=128, n_phonemes=42).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=args.probe_lr)
        
        for epoch in range(args.probe_epochs):
            probe.train()
            epoch_correct = 0
            epoch_valid = 0
            epoch_loss = 0
            n_batches = 0
            
            for features, labels in zip(train_feats, train_labels):
                logits = probe(features.to(device))
                logits_flat = logits.reshape(-1, 42)
                labels_flat = labels.reshape(-1).to(device)
                
                loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=0)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    preds = logits_flat.argmax(dim=-1)
                    valid = labels_flat != 0
                    if valid.any():
                        epoch_correct += (preds[valid] == labels_flat[valid]).sum().item()
                        epoch_valid += valid.sum().item()
                    epoch_loss += loss.item()
                    n_batches += 1
            
            train_acc = epoch_correct / max(epoch_valid, 1)
            
            # Validation
            val_acc, val_acc_active = evaluate_probe(probe, val_feats, val_labels, device)
            
            if (epoch + 1) % 2 == 0 or epoch == args.probe_epochs - 1:
                print(f"  Epoch {epoch+1}/{args.probe_epochs}: "
                      f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, "
                      f"Val Active Acc={val_acc_active:.4f}, Loss={epoch_loss/max(n_batches,1):.4f}")
        
        results[probe_name] = {'acc': val_acc, 'acc_active': val_acc_active}
    
    # ===== 7. 結果報告 =====
    print(f"\n{'='*60}")
    print(f"📊 Phoneme Probe Results")
    print(f"{'='*60}")
    print(f"  Checkpoint: {os.path.basename(args.checkpoint)}")
    print(f"  Probe epochs: {args.probe_epochs}")
    print(f"  Collected batches: {len(labels_list)}")
    print(f"")
    print(f"  {'Feature':30s} {'Overall':>10s} {'Active':>10s}")
    print(f"  {'-'*50}")
    for name, accs in results.items():
        print(f"  {name:30s} {accs['acc']*100:>9.2f}% {accs['acc_active']*100:>9.2f}%")
    
    ldn_acc = results.get("LDN (conv only)", {}).get('acc', 0)
    embs_acc = results.get("emb_s (LDN + gembed)", {}).get('acc', 0)
    
    print(f"\n📈 Analysis:")
    print(f"  Feature Magnitude: LDN={ldn_mag:.4f}, emb_s={embs_mag:.4f} (ratio={ldn_mag/(embs_mag+1e-8):.4f})")
    
    gap = embs_acc - ldn_acc
    if gap > 0.15:
        print(f"  gembed dominates phoneme recognition (+{gap*100:.1f}pp)")
        print(f"  → Aux CE has huge room to improve LDN")
    elif gap > 0.05:
        print(f"  gembed contributes moderately (+{gap*100:.1f}pp)")
        print(f"  → Aux CE can meaningfully improve LDN")
    else:
        print(f"  gembed barely helps phonemes (+{gap*100:.1f}pp)")
        print(f"  → gembed likely encodes speaker/semantic, not phonemes")
        print(f"  → Aux CE is critical for phoneme discrimination")
    
    # Save to file
    output_path = args.output_file if args.output_file else args.checkpoint.replace('.pth', '_probe_results.txt')
    
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_path, 'w') as f:
        f.write(f"Phoneme Probe Results\n")
        f.write(f"=====================\n")
        f.write(f"Checkpoint: {os.path.basename(args.checkpoint)}\n")
        f.write(f"Probe epochs: {args.probe_epochs}\n")
        f.write(f"Collected batches: {len(labels_list)}\n\n")
        f.write(f"{'Feature':30s} {'Overall':>10s} {'Active':>10s}\n")
        f.write(f"{'-'*50}\n")
        for name, accs in results.items():
            f.write(f"{name:30s} {accs['acc']*100:>9.2f}% {accs['acc_active']*100:>9.2f}%\n")
        
        f.write(f"\nAnalysis:\n")
        f.write(f"Feature Magnitude: LDN={ldn_mag:.4f}, emb_s={embs_mag:.4f} (ratio={ldn_mag/(embs_mag+1e-8):.4f})\n")
        if gap > 0.15:
            f.write(f"gembed dominates phoneme recognition (+{gap*100:.1f}pp)\n")
        elif gap > 0.05:
            f.write(f"gembed contributes moderately (+{gap*100:.1f}pp)\n")
        else:
            f.write(f"gembed barely helps phonemes (+{gap*100:.1f}pp)\n")
    print(f"\nResults saved to: {output_path}")

    print(f"\n  Reference: A3 w/o gembed (raw only) Aux CE acc ≈ 0.71")
    print(f"  If LDN baseline << 0.71, Aux CE has significant marginal value")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Phoneme Linear Probe")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--frame_labels_path', type=str, 
                        default='/padawan/frame_labels.lmdb',
                        help='Path to MFA frame labels LMDB')
    parser.add_argument('--probe_epochs', type=int, default=10,
                        help='Number of epochs to train linear probes')
    parser.add_argument('--probe_lr', type=float, default=0.0004,
                        help='Learning rate for probes')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size for feature collection')
    parser.add_argument('--n_collect_batches', type=int, default=100,
                        help='Number of batches to collect features from')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save results (default: checkpoint_name_probe_results.txt)')
    args = parser.parse_args()
    main(args)