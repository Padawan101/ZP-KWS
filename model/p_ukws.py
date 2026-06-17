"""
P-PhonMatchNet: Personalized User-defined Keyword Spotting

This module implements the complete P-PhonMatchNet model, integrating:
- Phase 1: SpeakerEncoder (frozen EfficientTDNN)
- Phase 2: FiLM conditioning (gamma, beta)
- Baseline: PhonMatchNet components (AudioEncoder, TextEncoder, etc.)

Architecture:
    Dual-path design (Y-shaped branching):
    - Path B (先執行): Speaker Verification
      Input Audio → SpeakerEncoder → cosine(E_input, E_enrollment) → P_spk

    - Path A (後執行): Keyword Spotting
      Input Audio → AudioEncoder → FiLM(speaker_emb) → modulated features
      Keyword Text → TextEncoder
      Concat → Self-Attention → GRU → P_utt

    - Fusion: score = P_utt × P_spk (Multiplicative)

Reference:
    - Spec v1.3 Section 2 (Architecture Design)
    - Spec v1.3 Section 3.4 (P_UKWS Implementation)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.dirname(__file__))
import encoder
import extractor
import discriminator
import log_melspectrogram
from model.utils import sequence_mask

# Import Phase 2 components (FiLM Module)
from film import FiLMGenerator
from film import FiLMLayer
from film import SoftConditionalFiLM  # v5.1: Soft Conditional FiLM
from film import EnhancedGatedFiLM   # v5.2: Enhanced Gated FiLM
from typing import Optional, Dict

# Import Auxiliary CE components (MFA frame-level supervision)
from losses import AuxPhonemeHead

# Import Phase 1 components - allow for mocking in tests
try:
    from speaker.encoder import SpeakerEncoder
except (ImportError, ModuleNotFoundError):
    # Allow tests to provide mock
    if 'model.speaker.encoder' in sys.modules:
        SpeakerEncoder = sys.modules['model.speaker.encoder'].SpeakerEncoder
    else:
        raise

seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)


class P_UKWS(nn.Module):
    """
    Personalized User-defined Keyword Spotting (P-PhonMatchNet)

    Extends BaseUKWS with speaker verification for personalized KWS.

    Architecture:
        - Path B (先執行): Speaker Verification
          Input Audio → SpeakerEncoder → cosine(E_input, E_enrollment) → P_spk

        - Path A (後執行): Keyword Spotting
          Input Audio → AudioEncoder → FiLM(speaker_emb) → modulated features
          Keyword Text → TextEncoder
          Concat → Self-Attention → GRU → P_utt

        - Fusion: score = P_utt × P_spk (Multiplicative)

    Args:
        speaker_encoder_path (str): Path to EfficientTDNN model checkpoint
        speaker_dim (int): Speaker embedding dimension (192 for EfficientTDNN Small)
        feature_dim (int): Audio/Text feature dimension (128)
        mode (str): KWS mode - affects how P_spk is used
            - "C-KWS": Conventional, only use P_utt (P_spk=1.0)
            - "TB-KWS": Target-user Biased, use P_utt × P_spk
            - "TO-KWS": Target-user Only, use P_utt × P_spk
        **kwargs: Arguments for baseline PhonMatchNet components
            - audio_input: 'both', 'raw', or 'google_embed'
            - text_input: 'g2p_embed' or 'phoneme'
            - stack_extractor: True for self-attention, False for cross-attention
            - frame_length, hop_length, num_mel, sample_rate, etc.

    Forward Args:
        speech: Speech features (format depends on audio_input mode)
        text: Text tokens [B, T_text]
        speech_len: Speech lengths
        text_len: Text lengths [B]
        enrollment_audio: Enrollment speech waveform [B, T_enroll] (optional)
        enrollment_emb: Pre-computed enrollment embedding [B, 192] (optional)

    Returns:
        dict containing:
            - 'P_utt': [B, 1] keyword match probability
            - 'P_spk': [B, 1] speaker match probability
            - 'score': [B, 1] final score (P_utt × P_spk)
            - 'prob': [B, 1] alias for score (for backward compatibility)
            - 'seq_ce_logit': [B, T_t] phoneme-level predictions (for L_phon loss)
            - 'affinity_matrix': attention weights
            - 'LD': latent discriminative features
            - 'affinity_mask': attention mask
            - 'seq_ce_logit_mask': phoneme mask

    Example:
        >>> # Initialize model
        >>> model = P_UKWS(
        ...     mode="TB-KWS",
        ...     audio_input='both',
        ...     text_input='g2p_embed',
        ...     stack_extractor=True,
        ...     vocab=42,
        ...     frame_length=400,
        ...     hop_length=160,
        ...     num_mel=40,
        ...     sample_rate=16000,
        ...     log_mel=True
        ... )
        >>>
        >>> # Prepare inputs
        >>> batch_size = 4
        >>> raw_audio = torch.randn(batch_size, 16000)
        >>> google_embed = torch.randn(batch_size, 100, 96)
        >>> speech = (raw_audio, google_embed)
        >>> speech_len = (torch.tensor([16000]*batch_size), torch.tensor([100]*batch_size))
        >>> text = torch.randn(batch_size, 50, 256)  # g2p_embed
        >>> text_len = torch.tensor([50]*batch_size)
        >>> enrollment_audio = torch.randn(batch_size, 16000)
        >>>
        >>> # Forward pass
        >>> output = model(
        ...     speech=speech,
        ...     text=text,
        ...     speech_len=speech_len,
        ...     text_len=text_len,
        ...     enrollment_audio=enrollment_audio
        ... )
        >>>
        >>> print(f"P_utt: {output['P_utt'].shape}")    # [4, 1]
        >>> print(f"P_spk: {output['P_spk'].shape}")    # [4, 1]
        >>> print(f"score: {output['score'].shape}")    # [4, 1]
    """

    def __init__(
        self,
        speaker_encoder_path: str = "model/speaker/efficient_tdnn",
        speaker_dim: int = 192,
        feature_dim: int = 128,
        mode: str = "TB-KWS",
        disable_film: bool = False,      # Ablation: 禁用 FiLM
        disable_sv_branch: bool = False, # Ablation: 禁用 SV branch
        freeze_speaker_encoder: bool = True,  # 是否凍結 Speaker Encoder
        finetuned_speaker_encoder_path: str = None,  # Finetuned checkpoint 路徑
        film_target: str = "fused",      # FiLM 作用目標: "fused" 或 "conv_only"
        film_gate_type: str = "pspk",    # v5.2: FiLM gate 類型
        # MFA Auxiliary CE (Section 2 of MFA_AuxCE_Design.md)
        enable_aux_ce: bool = False,     # 是否啟用 Auxiliary CE Loss
        n_phonemes: int = 42,            # 音素類別數 (42 = PAD + SIL + SPN + 39 ARPAbet)
        # A5: Gembed Feature Dropout
        gemb_drop_rate: float = 0.0,     # 訓練時 gembed dropout 機率 (0=off, 0.5=推薦)
        # A6: Curriculum Gembed
        gemb_curriculum: bool = False,   # 是否啟用漸進式引入 gembed
        gemb_warmup_epochs: int = 5,     # Phase 1: LDN-only epochs
        gemb_ramp_epochs: int = 10,      # Phase 2: 線性引入 gembed epochs
        # Calibration Layer Ablation
        calibration_mode: str = 'full',  # 'linear','raw_sigmoid','frozen_init','scale_only','bias_only','full'
        # Score Fusion Mode
        fusion_mode: str = 'multiply',   # 'multiply','harmonic','min'
        stream_fusion: str = 'add',       # 'add' (element-wise sum) | 'concat' (concat+projection)
        **kwargs
    ):
        """
        Initialize P-PhonMatchNet model

        Args:
            speaker_encoder_path: Path to EfficientTDNN model checkpoint
            speaker_dim: Speaker embedding dimension (192 for EfficientTDNN Small)
            feature_dim: Audio/Text feature dimension (128)
            mode: KWS mode ("C-KWS", "TB-KWS", "TO-KWS")
            finetuned_speaker_encoder_path: Path to finetuned checkpoint (optional)
            **kwargs: Arguments for baseline PhonMatchNet
        """
        super().__init__()

        # Validate mode
        assert mode in ["C-KWS", "TB-KWS", "TO-KWS"], \
            f"mode must be one of ['C-KWS', 'TB-KWS', 'TO-KWS'], got {mode}"

        self.mode = mode
        self.audio_input = kwargs['audio_input']
        self.text_input = kwargs['text_input']
        self.stack_extractor = kwargs['stack_extractor']
        self.speaker_dim = speaker_dim
        self.feature_dim = feature_dim

        # ===== Ablation Study 開關 =====
        self.disable_film = disable_film
        self.disable_sv_branch = disable_sv_branch
        
        # ===== FiLM Target 配置 =====
        assert film_target in ["fused", "conv_only"], \
            f"film_target must be 'fused' or 'conv_only', got {film_target}"
        self.film_target = film_target
        
        # ===== v5.2: FiLM Gate Type 配置 =====
        assert film_gate_type in ["pspk", "learned_scalar", "learned_channel"], \
            f"film_gate_type must be 'pspk', 'learned_scalar', or 'learned_channel', got {film_gate_type}"
        self.film_gate_type = film_gate_type
        
        if self.disable_film:
            print(">> [ABLATION] FiLM modulation DISABLED (γ=1, β=0)")
        else:
            print(f">> [FiLM] target={film_target}, gate_type={film_gate_type}")
            if film_gate_type != "pspk":
                gate_mode = "scalar" if film_gate_type == "learned_scalar" else "channel"
                print(f">>         Using EnhancedGatedFiLM with learned {gate_mode} gate (v5.2)")
            else:
                print(">>         Using SoftConditionalFiLM with P_spk gate (v5.1)")
            if film_target == "conv_only":
                print(">>         FiLM only modulates conv_feat, gembed kept unchanged")
        
        # ===== A5: Gembed Feature Dropout =====
        self.gemb_drop_rate = gemb_drop_rate
        if gemb_drop_rate > 0:
            print(f">> [Feature Dropout] gemb_drop_rate={gemb_drop_rate}")
            print(f">>   Training: {gemb_drop_rate*100:.0f}% samples use LDN only, {(1-gemb_drop_rate)*100:.0f}% use LDN+gembed")
            print(f">>   Inference: always LDN+gembed")
        
        # ===== A6: Curriculum Gembed =====
        self.gemb_curriculum = gemb_curriculum
        self.gemb_warmup_epochs = gemb_warmup_epochs
        self.gemb_ramp_epochs = gemb_ramp_epochs
        self._current_epoch = 0
        if gemb_curriculum:
            total = gemb_warmup_epochs + gemb_ramp_epochs
            print(f">> [Curriculum Gembed]")
            print(f">>   Phase 1 (α=0): epoch 0-{gemb_warmup_epochs-1}")
            print(f">>   Phase 2 (ramp): epoch {gemb_warmup_epochs}-{total-1}")
            print(f">>   Phase 3 (α=1): epoch {total}+")
        
        # ===== A5-Fix: LDN LayerNorm =====
        # 解決 LDN (magnitude~0.16) 與 gembed (magnitude~2.0) 的能量位階差異
        # LayerNorm 會將 LDN 標準化，然後通過 learnable γ/β 自適應到合適的量級
        if kwargs.get('audio_input', 'raw') in ('both', 'enhanced_gembed'):
            # DeConv-only mode (gru_layers=0) 不需要 LDN norm
            if kwargs.get('gru_layers', 2) == 0:
                print(">> [LDN Norm] Skipped (DeConv-only mode, no dual-stream)")
            elif kwargs.get('disable_ldn_norm', False):
                print(">> [ABLATION] LDN LayerNorm DISABLED")
            else:
                self.ldn_norm = nn.LayerNorm(feature_dim)
                print(f">> [LDN Norm] LayerNorm({feature_dim}) added for energy alignment with gembed")
        
        # ===== A2: Stream Fusion Mode =====
        self.stream_fusion = stream_fusion
        if stream_fusion == 'concat' and kwargs.get('audio_input', 'raw') in ('both', 'enhanced_gembed'):
            self.stream_fusion_proj = nn.Linear(feature_dim * 2, feature_dim)
            print(f">> [Stream Fusion] concat mode: Linear({feature_dim * 2} → {feature_dim})")
        elif stream_fusion == 'gated' and kwargs.get('audio_input', 'raw') in ('both', 'enhanced_gembed'):
            # A6: Gembed-Guided Gated Fusion
            self.ldn_gate = nn.Linear(feature_dim, feature_dim)
            # ★ 關鍵初始化：從 LayerNorm 的最優點啟動
            nn.init.zeros_(self.ldn_gate.weight)
            nn.init.constant_(self.ldn_gate.bias, -3.0)  # sigmoid(-3) ≈ 0.047
            self.stream_fusion_proj = None
            print(f">> [Stream Fusion] gated mode: gate = σ(Linear({feature_dim}→{feature_dim}))")
            print(f">>   Init: W=0, b=-3 → initial gate ≈ 0.047 (matches LN baseline)")
        else:
            self.stream_fusion_proj = None
            if kwargs.get('audio_input', 'raw') in ('both', 'enhanced_gembed'):
                print(f">> [Stream Fusion] add mode (default)")
        
        if self.disable_sv_branch:
            print(">> [ABLATION] SV branch DISABLED (P_spk=1, FiLM gate=0.5)")

        # ===== Phase 1: Speaker Encoder =====
        self.freeze_speaker_encoder = freeze_speaker_encoder
        
        # v5.2 FIX: auto-detect if speaker_encoder_path is a file (finetuned checkpoint)
        # If it's a file, we use the default base directory for initialization
        # and then load the weights.
        is_ckpt_file = os.path.isfile(speaker_encoder_path)
        if is_ckpt_file:
            print(f">> [SpeakerEncoder] Path is a file, assuming finetuned checkpoint: {speaker_encoder_path}")
            # Use default base path for architecture loading
            base_model_path = "model/speaker/efficient_tdnn" 
            print(f">> [SpeakerEncoder] Initializing architecture from: {base_model_path}")
            
            self.speaker_encoder = SpeakerEncoder(
                model_path=base_model_path,
                freeze=freeze_speaker_encoder
            )
            # Load the checkpoint
            self.speaker_encoder.load_finetuned_weights(speaker_encoder_path)
        else:
            # Standard directory initialization
            self.speaker_encoder = SpeakerEncoder(
                model_path=speaker_encoder_path,
                freeze=freeze_speaker_encoder
            )
        
        # 載入 finetuned weights（如果另外透過參數提供）
        # 如果 speaker_encoder_path 已經是 finetuned checkpoint，這裡可能會再次載入 (override)
        if finetuned_speaker_encoder_path is not None:
            self._load_finetuned_speaker_encoder(finetuned_speaker_encoder_path)
        
        if not freeze_speaker_encoder:
            print(">> [ABLATION] Speaker Encoder UNFROZEN (will be fine-tuned)")

        # Verify speaker encoder output dimension matches expected
        assert self.speaker_encoder.output_dim == speaker_dim, \
            f"Speaker encoder output dim {self.speaker_encoder.output_dim} != expected {speaker_dim}"
        
        # ===== Phase 2: FiLM Module =====
        # v5.2: 根據 film_gate_type 選擇 FiLM 模組
        if film_gate_type == "pspk":
            # v5.1 行為：使用 SoftConditionalFiLM
            self.soft_film = SoftConditionalFiLM(
                speaker_dim=speaker_dim,
                feature_dim=feature_dim
            )
            self.enhanced_film = None
        else:
            # v5.2 行為：使用 EnhancedGatedFiLM
            gate_type = "scalar" if film_gate_type == "learned_scalar" else "channel"
            self.enhanced_film = EnhancedGatedFiLM(
                speaker_dim=speaker_dim,
                feature_dim=feature_dim,
                gate_type=gate_type
            )
            self.soft_film = None
        
        # === Calibration Layer for Speaker Verification ===
        # Maps cosine similarity → P_spk probability
        # Mode controls which components are learnable (ablation study)
        assert calibration_mode in ('linear', 'raw_sigmoid', 'frozen_init',
                                    'scale_only', 'bias_only', 'full'), \
            f"calibration_mode must be one of ['linear','raw_sigmoid','frozen_init','scale_only','bias_only','full'], got {calibration_mode}"
        self.calibration_mode = calibration_mode
        self.max_spk_scale = 15.0  # Upper bound for spk_scale

        if calibration_mode == 'linear':
            # Cal-A: P_spk = (1 + cos) / 2, 無可學習參數
            print(f">> [Calibration] Mode: linear (Cal-A) - no learnable params")
        elif calibration_mode == 'raw_sigmoid':
            # Cal-B: P_spk = sigmoid(cos), 無可學習參數
            print(f">> [Calibration] Mode: raw_sigmoid (Cal-B) - no learnable params")
        elif calibration_mode == 'frozen_init':
            # Cal-C: P_spk = sigmoid(10*cos - 5), 凍結
            self.register_buffer('spk_scale', torch.tensor(10.0))
            self.register_buffer('spk_bias', torch.tensor(-5.0))
            print(f">> [Calibration] Mode: frozen_init (Cal-C) - w=10, b=-5 (frozen)")
        elif calibration_mode == 'scale_only':
            # Cal-D: P_spk = sigmoid(w*cos), 只有 scale 可學
            self.spk_scale = nn.Parameter(torch.tensor(10.0))
            self.register_buffer('spk_bias', torch.tensor(0.0))
            print(f">> [Calibration] Mode: scale_only (Cal-D) - w learnable, b=0 (fixed)")
        elif calibration_mode == 'bias_only':
            # Cal-E: P_spk = sigmoid(cos + b), 只有 bias 可學
            self.register_buffer('spk_scale', torch.tensor(10.0))
            self.spk_bias = nn.Parameter(torch.tensor(-7.0))
            print(f">> [Calibration] Mode: bias_only (Cal-E) - w=10 (fixed), b learnable")
        elif calibration_mode == 'full':
            # Cal-F: P_spk = sigmoid(w*cos + b), 都可學 (現有)
            self.spk_scale = nn.Parameter(torch.tensor(10.0))
            self.spk_bias = nn.Parameter(torch.tensor(-5.0))
            print(f">> [Calibration] Mode: full (Cal-F) - w, b both learnable")

        # ===== Score Fusion Mode =====
        assert fusion_mode in ('multiply', 'harmonic', 'min'), \
            f"fusion_mode must be 'multiply', 'harmonic', or 'min', got {fusion_mode}"
        self.fusion_mode = fusion_mode
        print(f">> [Fusion] Mode: {fusion_mode}")

        # ===== Baseline: Audio & Text Encoder =====
        embedding = feature_dim

        _stft = {
            'frame_length': kwargs['frame_length'],
            'hop_length': kwargs['hop_length'],
            'num_mel': kwargs['num_mel'],
            'sample_rate': kwargs['sample_rate'],
            'log_mel': kwargs['log_mel'],
            'lin_to_mel_path': "./model/lin_to_mel_matrix.npy",
        }

        if kwargs['audio_input'] == "google_embed":
            input_dim = 96
        else:
            input_dim = kwargs['num_mel']

        _ae = {
            'input_dim': input_dim,
            # [filter, kernel size, stride, padding]
            'conv': [[embedding, 5, 2, 2], [embedding * 2, 5, 1, 2]],
            # [unit]
            'gru': [[embedding] for _ in range(kwargs.get('gru_layers', 2))],
            # fully-connected layer unit
            'fc': embedding,
            'audio_input': self.audio_input,
            'bidirectional': kwargs.get('bidirectional', False),
        }

        _te = {
            # fully-connected layer unit
            'fc': embedding,
            # number of uniq. phonemes
            'vocab': kwargs['vocab'],
            'text_input': kwargs['text_input'],
        }

        _ext = {
            # [unit]
            'embedding': embedding,
            'num_heads': 1,
        }

        _dis = {
            'input_dim': embedding,
            # [unit]
            'gru': [[embedding],],
        }

        # Audio encoder
        if self.audio_input == 'both':  # two-stream audio encoder
            self.SPEC = log_melspectrogram.LogMelgramLayer(**_stft)
            if kwargs.get('disable_hybrid_encoder', False):
                print(">> [ABLATION] HybridAudioEncoder DISABLED -> Using EfficientAudioEncoder(downsample=False)")
                self.AE = encoder.EfficientAudioEncoder(downsample=False, **_ae)
                # Correction 2: efficient encoder does not support conv_only
                if self.film_target == 'conv_only':
                    print(">> [WARNING] EfficientAudioEncoder does not support 'conv_only' FiLM target. Falling back to 'fused'.")
                    self.film_target = 'fused'
            else:
                # A7: HybridAudioEncoder = AudioEncoder's conv+GRU+dense + gembed DeConv
                self.AE = encoder.HybridAudioEncoder(**_ae)
        elif self.audio_input == 'enhanced_gembed':
            # A3: Enhanced Gembed Encoder — 需要 SPEC 來產生 mask，但不處理 spectrogram
            self.SPEC = log_melspectrogram.LogMelgramLayer(**_stft)
            self.AE = encoder.EnhancedGembedEncoder(**_ae)
            print(">> [A3] Using EnhancedGembedEncoder: DeConv(T/2) + GRU")
        else:  # single-stream
            if self.audio_input == 'raw':
                self.FEAT = log_melspectrogram.LogMelgramLayer(**_stft)
            elif self.audio_input == 'google_embed':
                pass
            self.AE = encoder.AudioEncoder(**_ae)

        # Text encoder
        self.TE = encoder.TextEncoder(**_te)

        # Self-Attention / Cross-Attention
        if kwargs['stack_extractor']:
            self.EXT = extractor.StackExtractor(**_ext)  # self-attention
        else:
            self.EXT = extractor.BaseExtractor(**_ext)  # cross-attention

        # GRU discriminator for utterance-level prediction
        self.DIS = discriminator.BaseDiscriminator(**_dis)

        # Phoneme-level discriminator
        self.seq_ce_logit = nn.Linear(embedding, 1)

        # ===== MFA Auxiliary CE Head (Section 2.2) =====
        # 只用單層 Linear，位於 Audio Encoder 之後、FiLM 之前
        self.enable_aux_ce = enable_aux_ce
        if enable_aux_ce:
            self.aux_head = AuxPhonemeHead(
                feature_dim=feature_dim,
                n_phonemes=n_phonemes
            )
            print(f">> [Aux CE] Enabled with {n_phonemes} phoneme classes")
        else:
            self.aux_head = None

        # === FiLM 統計追蹤 ===
        self._last_gamma = None  # 最近一次 forward 的 gamma
        self._last_beta = None   # 最近一次 forward 的 beta
        self._last_gate = None   # v5.2: 最近一次 forward 的 learned gate

        # === v5.0: Soft Conditional FiLM Identity Init 驗證 ===
        # soft_film 在其 __init__ 中自動 identity init
        if hasattr(self, 'soft_film'):
            self._apply_film_identity_init()
    def compute_score(self, P_utt, P_spk):
        """
        Fuse P_utt and P_spk into final Score.

        Args:
            P_utt: [B] or [B, 1]  keyword match probability
            P_spk: [B] or [B, 1]  speaker match probability
        Returns:
            score: same shape as input
        """
        if self.fusion_mode == 'multiply':
            return P_utt * P_spk
        elif self.fusion_mode == 'harmonic':
            return 2.0 * P_utt * P_spk / (P_utt + P_spk + 1e-8)
        elif self.fusion_mode == 'min':
            return torch.min(P_utt, P_spk)

    def forward(
        self,
        speech: torch.Tensor,
        text: torch.Tensor,
        speech_len: torch.Tensor = None,
        text_len: torch.Tensor = None,
        enrollment_audio: torch.Tensor = None,
        enrollment_emb: torch.Tensor = None,
        same_speaker: torch.Tensor = None,  # 新增：[B] bool tensor for conditional FiLM
        verbose: bool = False,
        return_embeddings: bool = False,  # NEW: Return intermediate embeddings for t-SNE
        raw_audio_for_spk: torch.Tensor = None  # NEW: Raw audio for SV branch if speech is google_embed
    ) -> dict:
        """
        Forward pass for P-PhonMatchNet

        Args:
            speech: Speech features
                - if self.audio_input == 'both', shape - `((batch, time, mel), (batch, time/8, 96))`
                - elif self.audio_input == 'raw', shape - `(batch, time, mel)`
                - elif self.audio_input == 'google_embed', shape - `(batch, time/8, 96)`
            text: Text embedding of shape `(batch, phoneme)` or `(batch, phoneme, 256)` for g2p_embed
            speech_len: Length of speech parameter
                - if self.audio_input == 'both', shape - `((batch,), (batch,))`
                - else, shape - `(batch,)`
            text_len: Length of text parameter of shape `(batch,)`
            enrollment_audio: Enrollment speech waveform [B, T_enroll]
            enrollment_emb: Pre-computed enrollment embedding [B, 192]
            same_speaker: [B] bool tensor indicating if input audio is from same speaker as enrollment
                - Training: True = use FiLM, False = bypass FiLM
                - Inference: If None, always use FiLM
            verbose: Print debug information

        Returns:
            dict containing:
                - 'P_utt': [B, 1] keyword match probability
                - 'P_spk': [B, 1] speaker match probability
                - 'score': [B, 1] final score (P_utt × P_spk)
                - 'prob': [B, 1] alias for score
                - 'seq_ce_logit': [B, T_t] phoneme-level predictions
                - 'affinity_matrix': attention weights
                - 'LD': latent discriminative features
                - 'affinity_mask': attention mask
                - 'seq_ce_logit_mask': phoneme mask
        """
        # Get batch size
        if self.audio_input in ('both', 'enhanced_gembed'):
            batch_size = speech[0].size(0)
        else:
            batch_size = speech.size(0)

        # ===== Path B: Speaker Verification (先執行) =====
        
        # 取得 device
        device = speech[0].device if self.audio_input in ('both', 'enhanced_gembed') else speech.device
        
        # === v5.0: 簡化邏輯，移除 dropout ===
        # P_spk 作為 continuous gate，自動控制 FiLM 影響程度
        # 不再需要 batch-wise dropout
        
        # 決定是否使用 SV Branch
        if self.mode == "C-KWS" or self.disable_sv_branch:
            use_sv_branch = False  # C-KWS 或停用時不計算 P_spk
        else:
            use_sv_branch = True  # TB-KWS / TO-KWS：計算 P_spk
        
        # FiLM: C-KWS 不用
        use_film = (self.mode != "C-KWS") and (not self.disable_film)
        
        # === 檢查是否有 enrollment 可用 ===
        has_enrollment_audio = enrollment_audio is not None and (
            isinstance(enrollment_audio, torch.Tensor) and enrollment_audio.numel() > 0
        )
        has_enrollment_emb = enrollment_emb is not None and (
            isinstance(enrollment_emb, torch.Tensor) and enrollment_emb.numel() > 0
        )
        has_enrollment = has_enrollment_audio or has_enrollment_emb
        
        # === 計算 enrollment_emb（SV Branch 和 FiLM 都需要）===
        if has_enrollment and (use_sv_branch or use_film):
            if not has_enrollment_emb:
                with torch.no_grad():
                    enrollment_emb = self.speaker_encoder(enrollment_audio)
                    enrollment_emb = enrollment_emb.to(device)  # 確保在正確設備上
        elif use_film:
            # 需要 FiLM 但沒有 enrollment：使用 zero embedding
            enrollment_emb = torch.zeros(batch_size, self.speaker_dim, device=device)
            if verbose:
                print(f"[Warning] No enrollment provided, using zero embedding for FiLM")
        else:
            enrollment_emb = None
        
        # === 計算 P_spk（SV Branch）===
        if use_sv_branch and has_enrollment:
            # Get input embedding
            if raw_audio_for_spk is not None:
                input_audio_raw = raw_audio_for_spk
            elif self.audio_input in ('both', 'enhanced_gembed'):
                input_audio_raw = speech[0]  # [B, T]
            else:
                input_audio_raw = speech  # [B, T]
    
            with torch.no_grad():
                input_emb = self.speaker_encoder(input_audio_raw)
                input_emb = input_emb.to(device)  # 確保在正確設備上

            # 1. Compute cosine similarity
            cosine_score = F.cosine_similarity(
                input_emb, enrollment_emb, dim=1
            )  # [B]

            # 2. Apply calibration via compute_P_spk (supports ablation modes)
            P_spk = self.compute_P_spk(cosine_score)  # [B]

            # 3. Ensure correct shape [B, 1]
            if P_spk.dim() == 1:
                P_spk = P_spk.unsqueeze(-1)  # [B, 1]
        else:
            # C-KWS / disable_sv_branch / 無 enrollment：P_spk = 1.0
            P_spk = torch.ones(batch_size, 1, device=device)
            if verbose and use_sv_branch and not has_enrollment:
                print(f"[Validation Mode] No enrollment provided in {self.mode} mode")
                print(f"  → P_spk = 1.0 (fallback to C-KWS behavior)")

        # ===== Path A: Keyword Spotting (後執行) =====

        # Step 1: Audio encoding (same as baseline PhonMatchNet)
        gembed_processed = None  # 預設為 None，僅 conv_only 模式需要
        
        if self.audio_input in ('both', 'enhanced_gembed'):
            speech_raw, gemb = speech
            s_len, g_len = speech_len
            speech_processed, s_mask = self.SPEC(speech_raw, verbose)
            assert gemb.shape[-1] == 96, f"Google embedding should have 96 features, got {gemb.shape[-1]}"

            target_len = speech_processed.shape[1] // 8

            if gemb.shape[1] != target_len:
                if gemb.shape[1] > target_len:
                    # Truncate if too long
                    gemb = gemb[:, :target_len, :]
                else:
                    # Pad if too short
                    diff = target_len - gemb.shape[1]
                    gemb = F.pad(gemb, (0, 0, 0, diff), value=0.0)

            g_mask = sequence_mask(g_len, gemb.shape[1])
            
            if self.audio_input == 'enhanced_gembed':
                # A3: EnhancedGembedEncoder — single stream, no separation needed
                emb_s, LDN, emb_s_mask = self.AE(
                    (speech_processed, gemb), (s_mask, g_mask), verbose
                )
                gembed_processed = None
            else:
                # ★★★ 決定是否需要分離特徵 ★★★
                need_separate = (
                    # 原有條件：FiLM conv_only 需要分離
                    (not self.disable_film and enrollment_emb is not None and self.film_target == "conv_only")
                    # concat / gated fusion 需要分離
                    or self.stream_fusion == 'concat'
                    or self.stream_fusion == 'gated'
                )
                
                if need_separate:
                    emb_s, LDN, gembed_processed, emb_s_mask = self.AE(
                        (speech_processed, gemb), 
                        (s_mask, g_mask), 
                        verbose,
                        return_separate=True
                    )
                else:
                    emb_s, LDN, emb_s_mask = self.AE((speech_processed, gemb), (s_mask, g_mask), verbose)
                    gembed_processed = None  # add mode 不需要
        else:
            if self.audio_input == 'raw':
                speech_processed, s_mask = self.FEAT(speech, verbose)
            elif self.audio_input == 'google_embed':
                speech_processed = speech
                s_mask = sequence_mask(speech_len, speech.shape[1])

            emb_s, LDN, emb_s_mask = self.AE(speech_processed, s_mask, verbose)

        # ===== Stream Fusion + LDN Norm =====
        gemb_dropped = False
        gembed_magnitude = 0.0
        gembed_alpha = 1.0

        if self.audio_input == 'enhanced_gembed' and hasattr(self, 'ldn_norm'):
            # A3: 單一 stream，LDN 就是 emb_s 的 pre-activation
            # 不需要分離 gembed，不需要融合
            gembed_magnitude = 0.0  # 沒有分離的 gembed
            gembed_alpha = 1.0
            
            # 仍然對 LDN 做 LayerNorm（如果啟用），作為 regularization
            LDN = self.ldn_norm(LDN)
            
            # emb_s 已由 encoder 產生（post-activation），不需重新計算
            # 但需要用 normalized LDN 給 aux_head

        elif self.audio_input == 'both' and hasattr(self, 'ldn_norm'):

            if self.stream_fusion == 'gated' and gembed_processed is not None:
                # ---- A6: Gembed-Guided Gated Fusion ----
                gembed_magnitude = gembed_processed.abs().mean().item()

                # 1. Normalize LDN
                LDN = self.ldn_norm(LDN)

                # 2. Compute gate from gembed (content-adaptive)
                gate = torch.sigmoid(self.ldn_gate(gembed_processed))  # [B, T, 128]
                self._last_ldn_gate = gate.detach()  # 儲存用於 logging

                # 3. Gated fusion
                emb_s = gembed_processed + gate * LDN
                emb_s = torch.nan_to_num(emb_s) * emb_s_mask.unsqueeze(-1)

                # 4. 記錄 gate 統計
                gembed_alpha = gate.mean().item()

            elif self.stream_fusion == 'concat' and gembed_processed is not None:
                # ---- A2: Concat + Projection Mode ----
                # gembed_processed 已由 return_separate 取得，不需從 emb_s 反推
                gembed_magnitude = gembed_processed.abs().mean().item()

                # 1. Normalize LDN
                LDN = self.ldn_norm(LDN)

                # 2. Curriculum / Dropout（作用於 gembed_processed）
                if self.training and self.gemb_curriculum:
                    epoch = self._current_epoch
                    if epoch < self.gemb_warmup_epochs:
                        gembed_alpha = 0.0
                    elif epoch < self.gemb_warmup_epochs + self.gemb_ramp_epochs:
                        gembed_alpha = (epoch - self.gemb_warmup_epochs + 1) / self.gemb_ramp_epochs
                    else:
                        gembed_alpha = 1.0
                    gembed_input = gembed_alpha * gembed_processed
                elif self.training and self.gemb_drop_rate > 0:
                    gembed_alpha = 1.0
                    B_size = LDN.shape[0]
                    drop_mask = (torch.rand(B_size, 1, 1, device=LDN.device) < self.gemb_drop_rate)
                    gembed_input = torch.where(drop_mask, torch.zeros_like(gembed_processed), gembed_processed)
                    gemb_dropped = drop_mask.any().item()
                else:
                    gembed_alpha = 1.0
                    gembed_input = gembed_processed

                # 3. Activate LDN + Concat + Project
                LDN_act = F.leaky_relu(LDN, negative_slope=0.01)
                emb_s = self.stream_fusion_proj(
                    torch.cat([LDN_act, gembed_input], dim=-1)  # [B, T, 256] → [B, T, 128]
                )
                emb_s = torch.nan_to_num(emb_s) * emb_s_mask.unsqueeze(-1)

            else:
                # ---- Original: Add Mode ----
                gembed_part = emb_s - LDN
                gembed_magnitude = gembed_part.abs().mean().item()

                LDN = self.ldn_norm(LDN)

                if self.training and self.gemb_curriculum:
                    epoch = self._current_epoch
                    if epoch < self.gemb_warmup_epochs:
                        gembed_alpha = 0.0
                    elif epoch < self.gemb_warmup_epochs + self.gemb_ramp_epochs:
                        gembed_alpha = (epoch - self.gemb_warmup_epochs + 1) / self.gemb_ramp_epochs
                    else:
                        gembed_alpha = 1.0
                    emb_s = LDN + gembed_alpha * gembed_part
                elif self.training and self.gemb_drop_rate > 0:
                    gembed_alpha = 1.0
                    emb_s = LDN + gembed_part
                    B_size = emb_s.shape[0]
                    drop_mask = (torch.rand(B_size, 1, 1, device=emb_s.device) < self.gemb_drop_rate)
                    emb_s = torch.where(drop_mask, LDN, emb_s)
                    gemb_dropped = drop_mask.any().item()
                else:
                    gembed_alpha = 1.0
                    emb_s = LDN + gembed_part

        elif self.audio_input == 'both' and self.stream_fusion == 'concat' and gembed_processed is not None:
            # concat mode 但沒有 ldn_norm（disable_ldn_norm=True 的情況）
            gembed_magnitude = gembed_processed.abs().mean().item()
            LDN_act = F.leaky_relu(LDN, negative_slope=0.01)
            emb_s = self.stream_fusion_proj(
                torch.cat([LDN_act, gembed_processed], dim=-1)
            )
            emb_s = torch.nan_to_num(emb_s) * emb_s_mask.unsqueeze(-1)

        # ===== Auxiliary CE Head (Section 2.2) =====
        # 使用 normalized LDN（如果有 LayerNorm）
        # LDN 在上面的 LayerNorm 步驟後已經是 normalized 的
        aux_logits = None
        if self.training and self.enable_aux_ce and self.aux_head is not None:
            aux_logits = self.aux_head(LDN)  # [B, T_a, n_phonemes]

        # Step 2: FiLM Modulation (v5.1/v5.2)
        # P_spk 作為 continuous gate (v5.1) 或 hint (v5.2)

        if use_film and enrollment_emb is not None:
            
            # 決定 FiLM 輸入特徵
            if self.film_target == "conv_only" and gembed_processed is not None:
                film_input = LDN.transpose(1, 2)  # [B, 128, T]
                use_conv_only = True
            else:
                film_input = emb_s.transpose(1, 2)  # [B, 128, T]
                use_conv_only = False
            
            # 根據 film_gate_type 選擇 FiLM 模組
            if self.film_gate_type == "pspk":
                # v5.1 行為：P_spk 直接作為 gate
                if self.disable_sv_branch:
                    film_gate = torch.ones(batch_size, 1, device=device) * 0.5
                else:
                    film_gate = P_spk.detach()  # detach 防止 L_kws 梯度影響 spk_scale/spk_bias
                
                film_output, gamma, beta = self.soft_film(
                    film_input,
                    enrollment_emb,
                    film_gate
                )
                learned_gate = None  # 無 learned gate
            else:
                # v5.2 行為：Learned Gate (P_spk 作為 hint)
                film_output, gamma, beta, learned_gate = self.enhanced_film(
                    film_input,
                    enrollment_emb,
                    P_spk  # P_spk 作為 hint，內部會 detach
                )
            
            # 還原維度並組合
            film_output = film_output.transpose(1, 2)  # [B, T, 128]
            
            if use_conv_only:
                emb_s = film_output + gembed_processed
            else:
                emb_s = film_output
            
            # 儲存統計資訊
            self._last_gamma = gamma.detach()
            self._last_beta = beta.detach()
            self._last_gate = learned_gate.detach() if learned_gate is not None else None
        else:
            # disable_film / C-KWS 模式：不使用 FiLM
            self._last_gamma = None
            self._last_beta = None
            self._last_gate = None

        # Step 3: Text encoding
        emb_t, emb_t_mask = self.TE(text, verbose)

        # Step 4: Self-Attention / Cross-Attention
        attention_output, affinity_matrix, attention_mask, affinity_mask = \
            self.EXT(emb_s, emb_t, emb_s_mask, emb_t_mask, verbose)

        # Step 5: GRU discriminator for utterance-level prediction
        P_utt, LD = self.DIS(attention_output, attention_mask, verbose)

        # Step 6: Phoneme-level prediction (for L_phon loss)
        if self.stack_extractor:
            n_speech = torch.sum(emb_s_mask, dim=-1)
            n_text = torch.sum(emb_t_mask, dim=-1)
            n_total = n_speech + n_text
            # Masking only for the text part: [False, ..., False, True, ..., True, False, ...]
            valid_mask = torch.logical_xor(
                sequence_mask(n_total, max_length=attention_output.shape[1]),
                sequence_mask(n_speech, max_length=attention_output.shape[1])
            )
            indices = torch.masked_fill(torch.cumsum(valid_mask.int(), dim=1), ~valid_mask, 0)
            masked = torch.zeros(
                attention_output.shape[0],
                attention_output.shape[1] + 1,
                attention_output.shape[2]
            ).to(attention_output.device)
            masked = torch.scatter(
                input=masked,
                dim=1,
                index=torch.stack([indices for _ in range(attention_output.shape[-1])], dim=-1),
                src=attention_output
            )
            valid_attention_output = masked[:, 1:torch.max(n_text) + 1]
            seq_ce_logit = self.seq_ce_logit(valid_attention_output)[:, :, 0]
            seq_ce_logit = F.pad(seq_ce_logit, (0, emb_t.shape[1] - seq_ce_logit.shape[1]), value=0.)
            seq_ce_logit_mask = emb_t_mask
            seq_ce_logit = torch.nan_to_num(seq_ce_logit) * seq_ce_logit_mask
        else:
            seq_ce_logit = self.seq_ce_logit(attention_output)[:, :, 0]
            seq_ce_logit_mask = attention_mask
            seq_ce_logit = torch.nan_to_num(seq_ce_logit) * seq_ce_logit_mask

        # ===== Fusion =====
        score = self.compute_score(P_utt, P_spk)  # [B, 1]

        # Return all outputs
        result = {
            'P_utt': P_utt,                      # Keyword match probability
            'P_spk': P_spk,                      # Speaker match probability
            'score': score,                      # Final score (multiplicative fusion)
            'prob': score,                       # Alias for backward compatibility
            'seq_ce_logit': seq_ce_logit,        # Phoneme-level predictions
            'affinity_matrix': affinity_matrix,  # Attention weights
            'LD': LD,                            # Latent discriminative features
            'affinity_mask': affinity_mask,      # Attention mask
            'seq_ce_logit_mask': seq_ce_logit_mask,  # Phoneme mask
            # MFA Aux CE outputs (None if not enabled or not training)
            'aux_logits': aux_logits,            # [B, T_a, n_phonemes] frame-level logits
            'emb_s_mask': emb_s_mask,            # [B, T_a] audio encoder output mask
            # A5-Fix + A6: Feature Dropout / Curriculum monitoring
            'LDN_magnitude': LDN.abs().mean().item() if LDN is not None else 0.0,
            'gembed_magnitude': gembed_magnitude,  # gembed 的能量強度
            'gembed_alpha': gembed_alpha,           # A6: curriculum alpha (0→1) / gated: mean gate
            'gemb_dropped': gemb_dropped,           # A5: 是否有 sample 使用 LDN only
        }
        
        # NEW: Return intermediate embeddings for t-SNE visualization
        if return_embeddings:
            # Audio embedding (after FiLM modulation): mean-pool over time
            emb_s_pooled = (emb_s * emb_s_mask.unsqueeze(-1)).sum(dim=1) / emb_s_mask.sum(dim=1, keepdim=True).clamp(min=1)
            result['emb_audio'] = emb_s_pooled  # [B, feature_dim]
            
            # Text embedding: mean-pool over phonemes
            emb_t_pooled = (emb_t * emb_t_mask.unsqueeze(-1)).sum(dim=1) / emb_t_mask.sum(dim=1, keepdim=True).clamp(min=1)
            result['emb_text'] = emb_t_pooled  # [B, feature_dim]
            
            # Attention output (after self/cross attention): mean-pool
            attn_pooled = (attention_output * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
            result['emb_attention'] = attn_pooled  # [B, feature_dim]
            
            # Speaker embedding (if available)
            if enrollment_emb is not None:
                result['emb_speaker'] = enrollment_emb  # [B, speaker_dim]
        
        return result

    def compute_P_spk(self, cosine_sim: torch.Tensor) -> torch.Tensor:
        """
        根據 calibration_mode 計算 P_spk

        Args:
            cosine_sim: [B], cosine similarity ∈ [-1, 1]

        Returns:
            P_spk: [B], ∈ [0, 1]
        """
        if self.calibration_mode == 'linear':
            # Cal-A: (1 + cos) / 2
            P_spk = (1.0 + cosine_sim) / 2.0

        elif self.calibration_mode == 'raw_sigmoid':
            # Cal-B: sigmoid(cos)
            P_spk = torch.sigmoid(cosine_sim)

        else:
            # Cal-C/D/E/F: sigmoid(scale * cos + bias)
            # Clamp scale if learnable
            if self.calibration_mode in ('scale_only', 'full'):
                with torch.no_grad():
                    self.spk_scale.data.clamp_(min=3.0)

            scaled = self.spk_scale * cosine_sim + self.spk_bias
            P_spk = torch.sigmoid(scaled)

        return P_spk

    def set_current_epoch(self, epoch: int):
        """由 training loop 呼叫，更新 curriculum epoch，回傳當前 alpha"""
        self._current_epoch = epoch
        if self.gemb_curriculum:
            if epoch < self.gemb_warmup_epochs:
                alpha = 0.0
            elif epoch < self.gemb_warmup_epochs + self.gemb_ramp_epochs:
                alpha = (epoch - self.gemb_warmup_epochs + 1) / self.gemb_ramp_epochs
            else:
                alpha = 1.0
            return alpha
        return 1.0

    def set_mode(self, mode: str):
        """
        Switch between C-KWS, TB-KWS, TO-KWS modes

        Args:
            mode (str): One of ["C-KWS", "TB-KWS", "TO-KWS"]
                - "C-KWS": Conventional KWS, P_spk = 1.0
                - "TB-KWS": Target-user Biased KWS, score = P_utt × P_spk
                - "TO-KWS": Target-user Only KWS, score = P_utt × P_spk

        Example:
            >>> model = P_UKWS(mode="TB-KWS")
            >>> # Evaluate in conventional mode
            >>> model.set_mode("C-KWS")
            >>> output = model(speech, text, ...)
            >>> # Switch back to personalized mode
            >>> model.set_mode("TB-KWS")
        """
        assert mode in ["C-KWS", "TB-KWS", "TO-KWS"], \
            f"mode must be one of ['C-KWS', 'TB-KWS', 'TO-KWS'], got {mode}"
        self.mode = mode

    def freeze_speaker_encoder(self):
        """
        Ensure speaker encoder is frozen (no gradient updates)

        This is called automatically during __init__, but can be called
        again if needed to ensure speaker encoder remains frozen.
        """
        self.speaker_encoder.freeze()

    def get_trainable_parameters(self):
        """
        Get list of trainable parameters (excluding frozen speaker encoder)

        Returns:
            List of trainable parameters
        """
        trainable_params = []

        # v5.2: FiLM 模組 (trainable)
        if self.soft_film is not None:
            trainable_params.extend(self.soft_film.parameters())
        if self.enhanced_film is not None:
            trainable_params.extend(self.enhanced_film.parameters())

        # Calibration parameters (trainable)
        trainable_params.append(self.spk_scale)
        trainable_params.append(self.spk_bias)

        # Baseline PhonMatchNet components (trainable)
        if hasattr(self, 'SPEC'):
            trainable_params.extend(self.SPEC.parameters())
        if hasattr(self, 'FEAT'):
            trainable_params.extend(self.FEAT.parameters())
        trainable_params.extend(self.AE.parameters())
        trainable_params.extend(self.TE.parameters())
        trainable_params.extend(self.EXT.parameters())
        trainable_params.extend(self.DIS.parameters())
        trainable_params.extend(self.seq_ce_logit.parameters())

        # Aux CE Head (trainable, if enabled)
        if self.aux_head is not None:
            trainable_params.extend(self.aux_head.parameters())

        # Stream Fusion Projection (trainable, if concat mode)
        if self.stream_fusion_proj is not None:
            trainable_params.extend(self.stream_fusion_proj.parameters())

        return trainable_params

    def count_parameters(self, include_frozen: bool = False):
        """
        Count model parameters

        Args:
            include_frozen: If True, include frozen speaker encoder params

        Returns:
            dict with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in self.speaker_encoder.parameters())
        
        # v5.2: 支援兩種 FiLM 模組
        if self.soft_film is not None:
            film_params = sum(p.numel() for p in self.soft_film.parameters())
        elif self.enhanced_film is not None:
            film_params = sum(p.numel() for p in self.enhanced_film.parameters())
        else:
            film_params = 0

        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params,
            'film': film_params,
            'baseline': trainable_params - film_params
        }

    def get_film_stats(self) -> dict:
        """
        取得 FiLM 相關統計資訊，用於 TensorBoard 監控
        
        Returns:
            dict: 包含 gamma_mean, gamma_std, beta_mean, beta_std 等統計
        """
        stats = {}
        if hasattr(self, 'spk_scale'):
            stats['spk_scale'] = self.spk_scale.item()
        if hasattr(self, 'spk_bias'):
            stats['spk_bias'] = self.spk_bias.item()
        
        # v5.2: 支援兩種 FiLM 模組的權重統計
        if self.soft_film is not None:
            gamma_weight = self.soft_film.gamma_net[0].weight
            beta_weight = self.soft_film.beta_net.weight
            stats['film_gamma_weight_std'] = gamma_weight.std().item()
            stats['film_beta_weight_std'] = beta_weight.std().item()
        elif self.enhanced_film is not None:
            gamma_weight = self.enhanced_film.gamma_net[0].weight
            beta_weight = self.enhanced_film.beta_net.weight
            stats['film_gamma_weight_std'] = gamma_weight.std().item()
            stats['film_beta_weight_std'] = beta_weight.std().item()
        
        # === 從儲存的最近 forward 結果取得實際 gamma/beta ===
        if hasattr(self, '_last_gamma') and self._last_gamma is not None:
            stats['gamma_mean'] = self._last_gamma.mean().item()
            stats['gamma_std'] = self._last_gamma.std().item()
        else:
            stats['gamma_mean'] = 1.0
            stats['gamma_std'] = 0.0
            
        if hasattr(self, '_last_beta') and self._last_beta is not None:
            stats['beta_mean'] = self._last_beta.mean().item()
            stats['beta_std'] = self._last_beta.std().item()
        else:
            stats['beta_mean'] = 0.0
            stats['beta_std'] = 0.0
        
        return stats
    
    def get_gate_stats(self) -> Optional[Dict[str, float]]:
        """
        v5.2: 取得 Learned Gate 的統計資訊
        
        Returns:
            dict with gate_mean, gate_std, gate_min, gate_max
            或 None（如果使用 pspk 模式或無 gate 資料）
        """
        if not hasattr(self, '_last_gate') or self._last_gate is None:
            return None
        
        gate = self._last_gate
        return {
            'gate_mean': gate.mean().item(),
            'gate_std': gate.std().item(),
            'gate_min': gate.min().item(),
            'gate_max': gate.max().item(),
        }

    def _apply_film_identity_init(self):
        """
        v5.2: 驗證 FiLM Identity Initialization
        
        支援 soft_film 和 enhanced_film 兩種模式
        """
        print(">> [Init] Verifying FiLM Identity Initialization...")
        
        if self.soft_film is not None:
            # v5.1: 驗證 SoftConditionalFiLM
            with torch.no_grad():
                test_emb = torch.zeros(1, self.speaker_dim)
                test_audio = torch.ones(1, self.feature_dim, 10)
                test_pspk = torch.ones(1, 1)
                
                output, gamma, beta = self.soft_film(test_audio, test_emb, test_pspk)
                
                print(f"   -> [SoftConditionalFiLM] gamma_mean={gamma.mean().item():.4f}, beta_mean={beta.mean().item():.4f}")
                
                if abs(gamma.mean().item() - 1.0) > 0.01 or abs(beta.mean().item()) > 0.01:
                    print("!! [Warning] Identity init verification failed!")
                else:
                    print("   -> Identity init verified: gamma≈1, beta≈0 ✓")
        
        elif self.enhanced_film is not None:
            # v5.2: 驗證 EnhancedGatedFiLM
            with torch.no_grad():
                test_emb = torch.zeros(1, self.speaker_dim)
                test_audio = torch.ones(1, self.feature_dim, 10)
                test_pspk = torch.ones(1, 1)
                
                output, gamma, beta, gate = self.enhanced_film(test_audio, test_emb, test_pspk)
                
                print(f"   -> [EnhancedGatedFiLM] gamma_mean={gamma.mean().item():.4f}, beta_mean={beta.mean().item():.4f}, gate_mean={gate.mean().item():.4f}")
                
                if abs(gamma.mean().item() - 1.0) > 0.01 or abs(beta.mean().item()) > 0.01:
                    print("!! [Warning] FiLM identity init verification failed!")
                else:
                    print("   -> FiLM identity init verified: gamma≈1, beta≈0 ✓")
                
                # Gate 應該在 0.5 左右 (Sigmoid(0) = 0.5)
                if abs(gate.mean().item() - 0.5) > 0.1:
                    print(f"!! [Warning] Gate init unexpected: {gate.mean().item():.4f} (expected ~0.5)")
                else:
                    print(f"   -> Gate init verified: gate≈0.5 ✓")
        else:
            print("   -> No FiLM module found, skipping verification")

    def get_film_parameters(self):
        """
        v5.2: 取得 FiLM 模組參數
        
        Returns:
            list: FiLM 模組的所有參數
        """
        if self.soft_film is not None:
            return list(self.soft_film.parameters())
        elif self.enhanced_film is not None:
            return list(self.enhanced_film.parameters())
        return []
    
    def set_max_spk_scale(self, max_value: float):
        """
        Dynamic Scale Curriculum: 動態調整 spk_scale 的上限
        
        只在 calibration_mode 有可學習 scale 時有效。
        
        Args:
            max_value: 新的 spk_scale 上限值
        """
        if hasattr(self, 'spk_scale') and isinstance(self.spk_scale, nn.Parameter):
            self.max_spk_scale = max_value

    def get_non_film_parameters(self):
        """
        v3.0: 取得非 FiLM 的參數
        
        Returns:
            list: 除了 FiLM Generator 以外的所有可訓練參數
        """
        film_params = set(self.get_film_parameters())
        return [p for p in self.parameters() if p not in film_params and p.requires_grad]
    
    def _load_finetuned_speaker_encoder(self, checkpoint_path: str):
        """
        載入 finetuned speaker encoder checkpoint
        
        Args:
            checkpoint_path: Path to finetuned checkpoint (.pt file)
                Expected format: {'model_state_dict': ..., 'eer': ..., 'embedding_dim': ...}
        
        Note:
            The checkpoint is generated by speaker/finetune_speaker_encoder.py
            After loading, the speaker encoder will be frozen automatically.
        """
        import os
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Finetuned checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Load state dict
        self.speaker_encoder.encoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # Move to device
        device = next(self.parameters()).device
        self.speaker_encoder.encoder = self.speaker_encoder.encoder.to(device)
        
        # Log info
        eer = checkpoint.get('eer', 'N/A')
        epoch = checkpoint.get('epoch', 'N/A')
        print(f"✓ Loaded finetuned speaker encoder")
        print(f"  - Checkpoint: {checkpoint_path}")
        print(f"  - Epoch: {epoch}")
        print(f"  - EER: {eer:.2f}%" if isinstance(eer, (int, float)) else f"  - EER: {eer}")
        
        # Re-freeze parameters (important!)
        for param in self.speaker_encoder.encoder.parameters():
            param.requires_grad = False
        self.speaker_encoder.encoder.eval()
        print(f"  - Speaker encoder frozen after loading finetuned weights")


# ===== Testing Code =====
if __name__ == "__main__":
    print("=" * 80)
    print("P-PhonMatchNet Model Test")
    print("=" * 80)

    # Test configuration
    config = {
        'audio_input': 'both',
        'text_input': 'g2p_embed',
        'stack_extractor': True,
        'vocab': 42,
        'frame_length': 400,
        'hop_length': 160,
        'num_mel': 40,
        'sample_rate': 16000,
        'log_mel': True
    }

    print("\n1. Testing Model Initialization...")
    try:
        model = P_UKWS(mode="TB-KWS", **config)
        print("✓ Model initialized successfully")

        # Check components
        assert hasattr(model, 'speaker_encoder'), "Missing speaker_encoder"
        assert hasattr(model, 'film_generator'), "Missing film_generator"
        assert hasattr(model, 'film_layer'), "Missing film_layer"
        assert hasattr(model, 'AE'), "Missing AudioEncoder"
        assert hasattr(model, 'TE'), "Missing TextEncoder"
        assert hasattr(model, 'EXT'), "Missing Extractor"
        assert hasattr(model, 'DIS'), "Missing Discriminator"
        print("✓ All components present")

        # Parameter count
        param_counts = model.count_parameters()
        print(f"\nParameter counts:")
        print(f"  Total: {param_counts['total']:,}")
        print(f"  Trainable: {param_counts['trainable']:,}")
        print(f"  Frozen (SpeakerEncoder): {param_counts['frozen']:,}")
        print(f"  FiLM: {param_counts['film']:,}")
        print(f"  Baseline: {param_counts['baseline']:,}")

    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n2. Testing Forward Pass Shapes...")
    try:
        batch_size = 4

        # Prepare inputs
        raw_audio = torch.randn(batch_size, 16000)
        google_embed = torch.randn(batch_size, 100, 96)
        speech = (raw_audio, google_embed)
        speech_len = (torch.tensor([16000] * batch_size), torch.tensor([100] * batch_size))
        text = torch.randn(batch_size, 50, 256)  # g2p_embed
        text_len = torch.tensor([50] * batch_size)
        enrollment_audio = torch.randn(batch_size, 16000)

        # Forward pass
        output = model(
            speech=speech,
            text=text,
            speech_len=speech_len,
            text_len=text_len,
            enrollment_audio=enrollment_audio
        )

        # Check output shapes
        assert output['P_utt'].shape == (batch_size, 1), f"Expected P_utt shape {(batch_size, 1)}, got {output['P_utt'].shape}"
        assert output['P_spk'].shape == (batch_size, 1), f"Expected P_spk shape {(batch_size, 1)}, got {output['P_spk'].shape}"
        assert output['score'].shape == (batch_size, 1), f"Expected score shape {(batch_size, 1)}, got {output['score'].shape}"
        print(f"✓ P_utt shape: {output['P_utt'].shape}")
        print(f"✓ P_spk shape: {output['P_spk'].shape}")
        print(f"✓ score shape: {output['score'].shape}")

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n3. Testing Multiplicative Fusion...")
    try:
        # Verify score = P_utt * P_spk
        expected_score = output['P_utt'] * output['P_spk']
        diff = torch.abs(output['score'] - expected_score).max().item()
        assert diff < 1e-6, f"Fusion error: {diff}"
        print(f"✓ Multiplicative fusion correct (max error: {diff:.2e})")

    except Exception as e:
        print(f"✗ Fusion test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n4. Testing Mode Switching...")
    try:
        # Test C-KWS mode
        model.set_mode("C-KWS")
        output_c = model(
            speech=speech,
            text=text,
            speech_len=speech_len,
            text_len=text_len
        )
        assert torch.allclose(output_c['P_spk'], torch.ones_like(output_c['P_spk'])), \
            "C-KWS mode should have P_spk = 1.0"
        print(f"✓ C-KWS mode: P_spk = {output_c['P_spk'][0, 0]:.4f}")

        # Test TB-KWS mode
        model.set_mode("TB-KWS")
        output_tb = model(
            speech=speech,
            text=text,
            speech_len=speech_len,
            text_len=text_len,
            enrollment_audio=enrollment_audio
        )
        assert not torch.allclose(output_tb['P_spk'], torch.ones_like(output_tb['P_spk'])), \
            "TB-KWS mode should have varying P_spk"
        print(f"✓ TB-KWS mode: P_spk = {output_tb['P_spk'][0, 0]:.4f} (varies)")

        # Test TO-KWS mode
        model.set_mode("TO-KWS")
        output_to = model(
            speech=speech,
            text=text,
            speech_len=speech_len,
            text_len=text_len,
            enrollment_audio=enrollment_audio
        )
        print(f"✓ TO-KWS mode: P_spk = {output_to['P_spk'][0, 0]:.4f}")

    except Exception as e:
        print(f"✗ Mode switching failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n5. Testing Speaker Encoder is Frozen...")
    try:
        # Check that speaker encoder parameters don't require gradients
        for param in model.speaker_encoder.parameters():
            assert not param.requires_grad, "Speaker encoder should be frozen"
        print("✓ Speaker encoder is frozen (no gradients)")

    except Exception as e:
        print(f"✗ Freeze test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 80)
    print("All Tests Passed! ✓")
    print("=" * 80)
    print("\nP-PhonMatchNet model is ready for training!")
    print("\nNext steps:")
    print("  1. Modify dataset for speaker pairing")
    print("  2. Implement personalized evaluation metrics")
    print("  3. Create training script (train_personalized.py)")
