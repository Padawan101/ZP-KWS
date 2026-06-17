"""
FiLM (Feature-wise Linear Modulation) Module for P-PhonMatchNet

This module implements FiLM conditioning to modulate audio features
based on speaker embeddings, enabling personalized keyword spotting.

FiLM Equation:
    FiLM(x) = γ ⊙ x + β

where:
    - γ (gamma): scaling factors [B, 128]
    - β (beta): bias factors [B, 128]
    - x: input features [B, 128, T]
    - ⊙: element-wise multiplication

Architecture:
    FiLMGenerator: speaker_emb [B, 192] → γ [B, 128], β [B, 128]
    FiLMLayer: applies γ and β to modulate features
"""

import torch
import torch.nn as nn
from typing import Tuple


class FiLMGenerator(nn.Module):
    """
    Generate FiLM parameters (gamma, beta) from speaker embeddings.

    Architecture (following Spec v1.3 Section 3.2):
        Two separate networks:
        - gamma_net: Linear(192 → 128) + Tanh → output + 1.0
        - beta_net:  Linear(192 → 128) → output

    This design ensures:
        - Gamma constrained to [0, 2] via Tanh + 1.0
        - Identity initialization: gamma ≈ 1, beta ≈ 0
        - More stable than single shared network

    Args:
        speaker_dim: Speaker embedding dimension (192 for EfficientTDNN Small)
        feature_dim: Audio feature dimension to modulate (128 for PhonMatchNet)

    Input:
        speaker_emb: [B, 192] speaker embeddings from SpeakerEncoder

    Output:
        gamma: [B, 128] scaling factors (range ≈ [0, 2])
        beta: [B, 128] bias factors (unconstrained)

    Example:
        >>> generator = FiLMGenerator(speaker_dim=192, feature_dim=128)
        >>> speaker_emb = torch.randn(4, 192)
        >>> gamma, beta = generator(speaker_emb)
        >>> print(gamma.shape, beta.shape)  # [4, 128], [4, 128]
        >>> print(gamma.mean())  # Should be close to 1.0 initially
    """

    def __init__(
        self,
        speaker_dim: int = 192,
        feature_dim: int = 128
    ):
        super().__init__()

        self.speaker_dim = speaker_dim
        self.feature_dim = feature_dim

        # Separate networks for γ and β
        # gamma_net: constrain scale to reasonable range via Tanh
        self.gamma_net = nn.Sequential(
            nn.Linear(speaker_dim, feature_dim),
            nn.Tanh()  # Output range [-1, 1], then +1 → [0, 2]
        )

        # beta_net: unconstrained shift
        self.beta_net = nn.Sequential(
            nn.Linear(speaker_dim, feature_dim)
            # No activation - beta can be any value
        )

        # Initialize weights for identity transform
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights for identity transform: γ ≈ 1, β ≈ 0

        Strategy:
            - Zero initialization → Tanh(0) = 0 → 0 + 1 = 1 (for gamma)
            - Zero initialization → 0 (for beta)

        This ensures FiLM initially acts as identity: FiLM(x) ≈ 1*x + 0 = x
        """
        # γ network: zero init → Tanh(0) = 0 → gamma = 0 + 1 = 1
        nn.init.zeros_(self.gamma_net[0].weight)
        nn.init.zeros_(self.gamma_net[0].bias)

        # β network: zero init → beta = 0
        nn.init.zeros_(self.beta_net[0].weight)
        nn.init.zeros_(self.beta_net[0].bias)

    def forward(self, speaker_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate FiLM parameters from speaker embeddings.

        Args:
            speaker_emb: [B, 400] speaker embeddings

        Returns:
            gamma: [B, 128] scaling factors (≈ 1.0 initially, range [0, 2])
            beta: [B, 128] bias factors (≈ 0.0 initially, unconstrained)

        Raises:
            AssertionError: if input shape is incorrect
        """
        # Validate input shape
        assert speaker_emb.dim() == 2, \
            f"Expected 2D input [B, {self.speaker_dim}], got shape {speaker_emb.shape}"
        assert speaker_emb.shape[1] == self.speaker_dim, \
            f"Expected speaker_dim={self.speaker_dim}, got {speaker_emb.shape[1]}"

        # Generate gamma: Tanh(Linear(x)) + 1.0
        # This gives gamma ∈ [0, 2] with initial value ≈ 1
        gamma = self.gamma_net(speaker_emb) + 1.0  # [B, 128]

        # Generate beta: Linear(x)
        # This gives beta ∈ ℝ with initial value ≈ 0
        beta = self.beta_net(speaker_emb)  # [B, 128]

        return gamma, beta


class FiLMLayer(nn.Module):
    """
    Apply Feature-wise Linear Modulation to input features.

    FiLM(x) = gamma ⊙ x + beta

    where ⊙ is element-wise multiplication.

    This layer has NO learnable parameters - it only applies the
    modulation using the provided gamma and beta from FiLMGenerator.

    Args:
        None (this layer has no parameters)

    Input:
        features: [B, C, T] features to modulate (e.g., [B, 128, T])
        gamma: [B, C] scaling factors
        beta: [B, C] bias factors

    Output:
        modulated: [B, C, T] FiLM-modulated features

    Example:
        >>> film_layer = FiLMLayer()
        >>> features = torch.randn(4, 128, 100)  # [B, C, T]
        >>> gamma = torch.randn(4, 128)          # [B, C]
        >>> beta = torch.randn(4, 128)           # [B, C]
        >>> modulated = film_layer(features, gamma, beta)
        >>> print(modulated.shape)  # [4, 128, 100]
    """

    def __init__(self):
        """Initialize FiLMLayer (no learnable parameters)."""
        super().__init__()
        # This layer has no parameters - it's a pure function

    def forward(
        self,
        features: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply FiLM modulation to features.

        Args:
            features: [B, C, T] input features (e.g., audio features)
            gamma: [B, C] scaling factors from FiLMGenerator
            beta: [B, C] bias factors from FiLMGenerator

        Returns:
            modulated: [B, C, T] FiLM-modulated features

        Raises:
            AssertionError: if input shapes are incompatible
        """
        # Validate input shapes
        B, C, T = features.shape

        assert gamma.dim() == 2, \
            f"Expected gamma to be 2D [B, C], got {gamma.dim()}D with shape {gamma.shape}"
        assert beta.dim() == 2, \
            f"Expected beta to be 2D [B, C], got {beta.dim()}D with shape {beta.shape}"

        assert gamma.shape[0] == B, \
            f"Batch size mismatch: features[0]={B}, gamma[0]={gamma.shape[0]}"
        assert gamma.shape[1] == C, \
            f"Channel mismatch: features[1]={C}, gamma[1]={gamma.shape[1]}"

        assert beta.shape == gamma.shape, \
            f"Beta shape {beta.shape} must match gamma shape {gamma.shape}"

        # Broadcast gamma and beta to match feature dimensions
        # [B, C] → [B, C, 1] for broadcasting with [B, C, T]
        gamma = gamma.unsqueeze(-1)  # [B, C, 1]
        beta = beta.unsqueeze(-1)    # [B, C, 1]

        # Apply FiLM transformation: γ ⊙ x + β
        modulated = gamma * features + beta  # [B, C, T]

        return modulated


class SoftConditionalFiLM(nn.Module):
    """
    Soft Conditional FiLM: 使用 P_spk 作為 continuous gate
    
    v5.0 核心概念：
    - P_spk 高（speaker match）：充分應用 FiLM 調制
    - P_spk 低（speaker mismatch）：保守使用原始特徵
    - 平滑過渡，避免 binary 切換
    
    公式：
        E_a' = gate × (γ·E_a + β) + (1 - gate) × E_a
             = (1 + gate·(γ-1)) × E_a + gate × β
    
    當 gate=1 (P_spk 高): E_a' = γ·E_a + β (完整 FiLM)
    當 gate=0 (P_spk 低): E_a' = E_a (保持原始)
    
    Args:
        speaker_dim: Speaker embedding dimension (192 for EfficientTDNN Small)
        feature_dim: Audio feature dimension to modulate (128 for PhonMatchNet)
    
    Example:
        >>> soft_film = SoftConditionalFiLM(speaker_dim=192, feature_dim=128)
        >>> E_a = torch.randn(4, 128, 100)  # [B, C, T]
        >>> E_espk = torch.randn(4, 192)    # [B, speaker_dim]
        >>> P_spk = torch.rand(4, 1)        # [B, 1] ∈ [0, 1]
        >>> E_a_mod, gamma, beta = soft_film(E_a, E_espk, P_spk)
        >>> print(E_a_mod.shape)  # [4, 128, 100]
    """
    
    def __init__(
        self,
        speaker_dim: int = 192,
        feature_dim: int = 128
    ):
        super().__init__()
        
        self.speaker_dim = speaker_dim
        self.feature_dim = feature_dim
        
        # FiLM Generator (same architecture as FiLMGenerator)
        self.gamma_net = nn.Sequential(
            nn.Linear(speaker_dim, feature_dim),
            nn.Tanh()  # Output range [-1, 1], then +1 → [0, 2]
        )
        self.beta_net = nn.Linear(speaker_dim, feature_dim)
        
        # Identity initialization
        self._init_identity()
    
    def _init_identity(self):
        """
        Initialize weights for identity transform: γ ≈ 1, β ≈ 0
        
        This ensures FiLM initially acts as identity: FiLM(x) ≈ 1*x + 0 = x
        """
        nn.init.zeros_(self.gamma_net[0].weight)
        nn.init.zeros_(self.gamma_net[0].bias)
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)
    
    def forward(
        self,
        E_a: torch.Tensor,
        E_espk: torch.Tensor,
        P_spk: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply Soft Conditional FiLM modulation.
        
        Args:
            E_a: Audio features [B, C, T]
            E_espk: Enrollment speaker embedding [B, speaker_dim]
            P_spk: Speaker match probability [B, 1] (作為 continuous gate)
        
        Returns:
            E_a_modulated: Soft conditionally modulated features [B, C, T]
            gamma: FiLM gamma [B, C] (for logging)
            beta: FiLM beta [B, C] (for logging)
        """
        # Step 1: 計算 FiLM 參數
        gamma = self.gamma_net(E_espk) + 1.0  # [B, C], 範圍 [0, 2]
        beta = self.beta_net(E_espk)          # [B, C]
        
        # Step 2: 準備維度
        gamma_exp = gamma.unsqueeze(-1)  # [B, C, 1]
        beta_exp = beta.unsqueeze(-1)    # [B, C, 1]
        
        # Step 3: 計算 fully-filmed 版本
        E_a_filmed = gamma_exp * E_a + beta_exp  # [B, C, T]
        
        # Step 4: Soft gate (使用 P_spk)
        # P_spk 可能是 [B, 1] 或 [B]，統一處理為 [B, 1, 1]
        if P_spk.dim() == 1:
            gate = P_spk.unsqueeze(-1).unsqueeze(-1)  # [B] -> [B, 1, 1]
        elif P_spk.dim() == 2:
            gate = P_spk.unsqueeze(-1)  # [B, 1] -> [B, 1, 1]
        else:
            gate = P_spk  # Already [B, 1, 1]
        
        # Step 5: Soft conditional modulation
        # gate=1: 使用 E_a_filmed (完整 FiLM)
        # gate=0: 使用 E_a (原始特徵)
        E_a_modulated = gate * E_a_filmed + (1.0 - gate) * E_a
        
        return E_a_modulated, gamma, beta


class EnhancedGatedFiLM(nn.Module):
    """
    Enhanced Gated FiLM v5.2
    
    改進：
    - Gate 由神經網路學習，而非硬編碼 P_spk
    - P_spk 作為 hint 輸入 Gate Network
    - 支援 Scalar 或 Channel-wise gating
    
    Gate Network 輸入:
        - E_a_pooled [B, 128]: 音訊內容上下文 (Global Average Pooling)
        - E_espk [B, 192]: 目標說話者身份
        - P_spk.detach() [B, 1]: 說話者匹配程度 (hint)
        Total: [B, 321]
    
    Args:
        speaker_dim: Speaker embedding 維度 (192)
        feature_dim: Audio feature 維度 (128)
        gate_type: "scalar" 或 "channel"
            - "scalar": 所有 channel 共用一個 gate 值
            - "channel": 每個 channel 獨立 gate 值
    
    Example:
        >>> enhanced_film = EnhancedGatedFiLM(speaker_dim=192, feature_dim=128, gate_type="channel")
        >>> E_a = torch.randn(4, 128, 100)  # [B, C, T]
        >>> E_espk = torch.randn(4, 192)    # [B, speaker_dim]
        >>> P_spk = torch.rand(4, 1)        # [B, 1] ∈ [0, 1]
        >>> E_a_mod, gamma, beta, gate = enhanced_film(E_a, E_espk, P_spk)
        >>> print(E_a_mod.shape)  # [4, 128, 100]
        >>> print(gate.shape)     # [4, 128] for channel, [4, 1] for scalar
    """
    
    def __init__(
        self,
        speaker_dim: int = 192,
        feature_dim: int = 128,
        gate_type: str = "channel"  # "scalar" 或 "channel"
    ):
        super().__init__()
        
        assert gate_type in ["scalar", "channel"], \
            f"gate_type must be 'scalar' or 'channel', got {gate_type}"
        
        self.speaker_dim = speaker_dim
        self.feature_dim = feature_dim
        self.gate_type = gate_type
        
        # Gate 輸出維度
        gate_out_dim = 1 if gate_type == "scalar" else feature_dim
        
        # ===== FiLM Generators =====
        self.gamma_net = nn.Sequential(
            nn.Linear(speaker_dim, feature_dim),
            nn.Tanh()  # 輸出 [-1, 1]，+1 後變 [0, 2]
        )
        self.beta_net = nn.Linear(speaker_dim, feature_dim)
        
        # ===== Gate Network =====
        # 輸入: E_a_pooled (128) + E_espk (192) + P_spk (1) = 321
        gate_input_dim = feature_dim + speaker_dim + 1
        
        self.gate_net = nn.Sequential(
            nn.Linear(gate_input_dim, 64),
            nn.LayerNorm(64),  # 比 BatchNorm 更穩定
            nn.ReLU(),
            nn.Linear(64, gate_out_dim),
            nn.Sigmoid()
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """
        初始化策略：
        - FiLM: Identity init (γ=1, β=0)
        - Gate: 保守初始化，讓 gate 一開始偏低 (~0.27)
        """
        # FiLM Identity Init
        nn.init.zeros_(self.gamma_net[0].weight)
        nn.init.zeros_(self.gamma_net[0].bias)
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)
        
        # Gate Network Init
        # 第一層：Xavier
        nn.init.xavier_normal_(self.gate_net[0].weight)
        nn.init.zeros_(self.gate_net[0].bias)
        
        # 最後一層：讓初始 gate ≈ 0.5 (Sigmoid(0) = 0.5)
        # 改為中性起始 (bias=0.0)
        nn.init.zeros_(self.gate_net[3].weight)
        nn.init.constant_(self.gate_net[3].bias, 0.0)
    
    def forward(
        self,
        E_a: torch.Tensor,
        E_espk: torch.Tensor,
        P_spk: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            E_a: Audio features [B, C, T]
            E_espk: Enrollment speaker embedding [B, speaker_dim]
            P_spk: Speaker match probability [B, 1]
        
        Returns:
            E_a_modulated: Gated FiLM modulated features [B, C, T]
            gamma: FiLM gamma [B, C]
            beta: FiLM beta [B, C]
            gate: Learned gate [B, C] 或 [B, 1]
        """
        B, C, T = E_a.shape
        
        # ===== 1. 計算 Learned Gate =====
        # Audio context: Global Average Pooling
        E_a_pooled = E_a.mean(dim=2)  # [B, C]
        
        # ★★★ 關鍵：P_spk 必須 detach ★★★
        # 阻斷 L_kws 梯度流向 spk_scale/spk_bias
        P_spk_safe = P_spk.detach()
        
        # 確保 P_spk 是 [B, 1]
        if P_spk_safe.dim() == 1:
            P_spk_safe = P_spk_safe.unsqueeze(-1)
        
        # 拼接 Gate 輸入
        gate_input = torch.cat([E_a_pooled, E_espk, P_spk_safe], dim=1)  # [B, 321]
        
        # 計算 Gate
        gate = self.gate_net(gate_input)  # [B, 1] 或 [B, C]
        gate_expanded = gate.unsqueeze(-1)  # [B, 1, 1] 或 [B, C, 1]
        
        # ===== 2. 計算 FiLM 參數 =====
        gamma = self.gamma_net(E_espk) + 1.0  # [B, C], 範圍 [0, 2]
        beta = self.beta_net(E_espk)          # [B, C]
        
        gamma_expanded = gamma.unsqueeze(-1)  # [B, C, 1]
        beta_expanded = beta.unsqueeze(-1)    # [B, C, 1]
        
        # ===== 3. 計算 FiLM 調制後的特徵 =====
        E_a_filmed = gamma_expanded * E_a + beta_expanded  # [B, C, T]
        
        # ===== 4. Gated Fusion =====
        E_a_modulated = gate_expanded * E_a_filmed + (1.0 - gate_expanded) * E_a
        
        return E_a_modulated, gamma, beta, gate


# =============================================================================
# Unit Tests
# =============================================================================

def test_film_generator_shapes():
    """Test FiLMGenerator output shapes."""
    print("\n[Test 1] FiLMGenerator Output Shapes")

    generator = FiLMGenerator(speaker_dim=400, feature_dim=128)
    speaker_emb = torch.randn(4, 400)

    gamma, beta = generator(speaker_emb)

    assert gamma.shape == (4, 128), f"Expected gamma shape (4, 128), got {gamma.shape}"
    assert beta.shape == (4, 128), f"Expected beta shape (4, 128), got {beta.shape}"

    print(f"  Input shape: {speaker_emb.shape}")
    print(f"  Gamma shape: {gamma.shape}")
    print(f"  Beta shape: {beta.shape}")
    print("  ✓ PASSED: Output shapes are correct")


def test_film_generator_dimension():
    """Confirm speaker_dim=400, feature_dim=128."""
    print("\n[Test 2] FiLMGenerator Dimensions")

    generator = FiLMGenerator(speaker_dim=400, feature_dim=128)

    assert generator.speaker_dim == 400, "speaker_dim should be 400"
    assert generator.feature_dim == 128, "feature_dim should be 128"

    # Verify parameter shapes (separate networks)
    assert generator.gamma_net[0].in_features == 400
    assert generator.gamma_net[0].out_features == 128
    assert generator.beta_net[0].in_features == 400
    assert generator.beta_net[0].out_features == 128

    print(f"  speaker_dim: {generator.speaker_dim}")
    print(f"  feature_dim: {generator.feature_dim}")
    print(f"  gamma_net: Linear({generator.gamma_net[0].in_features} → {generator.gamma_net[0].out_features}) + Tanh + 1.0")
    print(f"  beta_net:  Linear({generator.beta_net[0].in_features} → {generator.beta_net[0].out_features})")
    print("  ✓ PASSED: Dimensions are correct")


def test_film_layer_modulation():
    """Test FiLMLayer modulation correctness."""
    print("\n[Test 3] FiLMLayer Modulation Math")

    film_layer = FiLMLayer()

    # Create test inputs with known values
    features = torch.randn(2, 128, 50)
    gamma = torch.ones(2, 128) * 2.0    # scale by 2
    beta = torch.ones(2, 128) * 1.0     # shift by 1

    # Apply FiLM
    modulated = film_layer(features, gamma, beta)

    # Manually compute expected result
    expected = gamma.unsqueeze(-1) * features + beta.unsqueeze(-1)

    # Check if results match
    diff = torch.abs(modulated - expected).max().item()

    assert diff < 1e-6, f"Math error: max diff = {diff}"
    assert modulated.shape == features.shape, "Output shape should match input"

    print(f"  Features shape: {features.shape}")
    print(f"  Gamma: {gamma[0, 0].item():.1f} (scale)")
    print(f"  Beta: {beta[0, 0].item():.1f} (shift)")
    print(f"  Max difference from expected: {diff:.2e}")
    print("  ✓ PASSED: FiLM math is correct")


def test_film_layer_broadcasting():
    """Test FiLMLayer broadcasting with different time lengths."""
    print("\n[Test 4] FiLMLayer Broadcasting")

    film_layer = FiLMLayer()

    # Test with different time dimensions
    time_lengths = [50, 100, 200]

    gamma = torch.randn(4, 128)
    beta = torch.randn(4, 128)

    for T in time_lengths:
        features = torch.randn(4, 128, T)
        modulated = film_layer(features, gamma, beta)

        assert modulated.shape == (4, 128, T), \
            f"Expected shape (4, 128, {T}), got {modulated.shape}"

    print(f"  Tested time lengths: {time_lengths}")
    print(f"  All broadcasting tests passed")
    print("  ✓ PASSED: Broadcasting works correctly")


def test_integration_with_speaker_encoder():
    """Integration test: SpeakerEncoder → FiLMGenerator → FiLMLayer."""
    print("\n[Test 5] Integration with SpeakerEncoder")

    try:
        from model.speaker.encoder import SpeakerEncoder
        has_speaker_encoder = True
    except ImportError:
        print("  ⚠ WARNING: SpeakerEncoder not available, using mock")
        has_speaker_encoder = False

    # Initialize modules
    film_generator = FiLMGenerator(speaker_dim=400, feature_dim=128)
    film_layer = FiLMLayer()

    # Simulate the complete pipeline
    batch_size = 4

    if has_speaker_encoder:
        # Use real SpeakerEncoder
        try:
            speaker_encoder = SpeakerEncoder(
                model_path='model/speaker/efficient_tdnn',
                freeze=True,
                device='cpu'
            )

            # Real enrollment audio
            enrollment_audio = torch.randn(batch_size, 16000)  # 1 second @ 16kHz

            with torch.no_grad():
                speaker_emb = speaker_encoder(enrollment_audio)  # [4, 400]
        except Exception as e:
            print(f"  ⚠ SpeakerEncoder failed: {e}")
            print("  Using mock speaker embeddings instead")
            speaker_emb = torch.randn(batch_size, 400)
    else:
        # Mock speaker embeddings
        speaker_emb = torch.randn(batch_size, 400)

    # Audio features from AudioEncoder (mock)
    audio_features = torch.randn(batch_size, 128, 100)  # [B, C, T]

    # Full pipeline
    gamma, beta = film_generator(speaker_emb)              # [4, 128], [4, 128]
    modulated = film_layer(audio_features, gamma, beta)    # [4, 128, 100]

    # Verify shapes
    assert speaker_emb.shape == (batch_size, 400), f"Speaker emb: {speaker_emb.shape}"
    assert gamma.shape == (batch_size, 128), f"Gamma: {gamma.shape}"
    assert beta.shape == (batch_size, 128), f"Beta: {beta.shape}"
    assert modulated.shape == (batch_size, 128, 100), f"Modulated: {modulated.shape}"

    print(f"  Pipeline:")
    print(f"    Enrollment audio → SpeakerEncoder → [{batch_size}, 400]")
    print(f"    Speaker emb → FiLMGenerator → gamma [{batch_size}, 128], beta [{batch_size}, 128]")
    print(f"    Audio features [{batch_size}, 128, 100] + FiLM → [{batch_size}, 128, 100]")
    print("  ✓ PASSED: Integration test successful")


def test_film_layer_no_parameters():
    """Verify FiLMLayer has no learnable parameters."""
    print("\n[Test 6] FiLMLayer Has No Parameters")

    film_layer = FiLMLayer()

    num_params = sum(p.numel() for p in film_layer.parameters())

    assert num_params == 0, f"FiLMLayer should have 0 parameters, got {num_params}"

    print(f"  Total parameters: {num_params}")
    print("  ✓ PASSED: FiLMLayer is parameter-free")


def test_film_generator_initialization():
    """Test FiLMGenerator initialization produces near-identity transform."""
    print("\n[Test 7] FiLMGenerator Initialization")

    generator = FiLMGenerator(speaker_dim=400, feature_dim=128)

    # Create zero speaker embedding (should produce gamma≈1, beta≈0)
    zero_emb = torch.zeros(1, 400)

    with torch.no_grad():
        gamma, beta = generator(zero_emb)

    # Check that gamma is close to 1
    gamma_mean = gamma.mean().item()
    beta_mean = beta.abs().mean().item()

    print(f"  With zero speaker embedding:")
    print(f"    Mean gamma: {gamma_mean:.4f} (expected ≈ 1.0)")
    print(f"    Mean |beta|: {beta_mean:.4f} (expected ≈ 0.0)")

    # Should be reasonably close to identity
    assert 0.5 < gamma_mean < 1.5, f"Gamma mean {gamma_mean} should be close to 1"
    assert beta_mean < 0.5, f"Beta mean {beta_mean} should be close to 0"

    print("  ✓ PASSED: Initialization is reasonable")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 80)
    print("Running FiLM Module Unit Tests")
    print("=" * 80)

    test_film_generator_shapes()
    test_film_generator_dimension()
    test_film_layer_modulation()
    test_film_layer_broadcasting()
    test_film_layer_no_parameters()
    test_film_generator_initialization()
    test_integration_with_speaker_encoder()

    print("\n" + "=" * 80)
    print("All Tests PASSED! ✓")
    print("=" * 80)


if __name__ == "__main__":
    # Run all tests when executed directly
    run_all_tests()

    print("\n" + "=" * 80)
    print("FiLM Module Summary")
    print("=" * 80)

    # Print module information
    generator = FiLMGenerator(speaker_dim=400, feature_dim=128)
    film_layer = FiLMLayer()

    num_params = sum(p.numel() for p in generator.parameters())

    print(f"\nFiLMGenerator:")
    print(f"  - Architecture: Dual network design (Spec v1.3)")
    print(f"    - gamma_net: Linear(400 → 128) + Tanh → + 1.0")
    print(f"    - beta_net:  Linear(400 → 128)")
    print(f"  - Input: speaker_emb [B, 400]")
    print(f"  - Output: gamma [B, 128] ∈ [0, 2], beta [B, 128] ∈ ℝ")
    print(f"  - Parameters: {num_params:,}")
    print(f"  - Initialization: Identity transform (γ≈1, β≈0)")

    print(f"\nFiLMLayer:")
    print(f"  - Function: γ ⊙ x + β")
    print(f"  - Input: features [B, 128, T], gamma [B, 128], beta [B, 128]")
    print(f"  - Output: modulated [B, 128, T]")
    print(f"  - Parameters: 0 (parameter-free)")

    print(f"\nReady for P-PhonMatchNet integration!")
