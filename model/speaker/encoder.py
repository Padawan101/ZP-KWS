"""
Speaker Encoder Module for P-PhonMatchNet

This module wraps EfficientTDNN Small (899.20K params) for speaker verification.
The encoder is frozen and used only for feature extraction.

Architecture: EfficientTDNN Small
- Depth: 2 layers
- Channels: [256, 256, 256]
- Kernels: [3, 3, 3]
- Output: 192-d speaker embedding (auto-detected from pretrained weights)
- Parameters: 899.20K (frozen)

"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Handle sugar import path
# sugar is NOT a pip package - must be added to sys.path
sugar_path = Path(__file__).parent / 'sugar'
if str(sugar_path) not in sys.path:
    sys.path.insert(0, str(sugar_path))

# Import only what we need to avoid dependency issues
try:
    from sugar.models.dynamictdnn import tdnn8m2g
    from sugar.transforms import LogMelFbanks
except ImportError as e:
    print(f"Warning: Could not import sugar components: {e}")
    print("Attempting to use minimal sugar imports...")
    # Fallback to basic import
    pass


class SpeakerEncoder(nn.Module):
    """
    EfficientTDNN Small Speaker Encoder

    This encoder extracts 192-dimensional speaker embeddings from raw audio waveforms.
    All parameters are frozen by default - this module is used only for feature extraction.

    Args:
        model_path: Path to local model directory containing:
            - width2.torchparams (supernet weights)
            - width2.2.256.256.256.3.3.3.400.bn.tar (subnet config)
        freeze: Whether to freeze all parameters (default: True)
        device: Device to load model on ('cuda' or 'cpu')

    Input:
        waveform: [B, T] @ 16kHz, raw audio waveform

    Output:
        embedding: [B, output_dim] speaker embedding (dimension auto-detected, typically 192)

    Example:
        >>> encoder = SpeakerEncoder(device='cuda', freeze=True)
        >>> waveform = torch.randn(4, 16000).cuda()  # 4 samples, 1 second
        >>> embedding = encoder(waveform)  # [4, 192]
    """

    def __init__(
        self,
        model_path: str = 'model/speaker/efficient_tdnn',
        freeze: bool = True,
        device: str = 'cuda'
    ):
        super().__init__()

        self.device = device
        # Note: output_dim will be set after loading subnet (could be 192 or 400)

        # Load model
        print(f"Loading EfficientTDNN Small from {model_path}...")

        # Construct paths to required files
        supernet_path = Path(model_path) / 'width2.torchparams'
        subnet_path = Path(model_path) / 'width2.2.256.256.256.3.3.3.400.bn.tar'

        # Verify files exist
        if not supernet_path.exists() or not subnet_path.exists():
            raise FileNotFoundError(
                f"Model files not found in {model_path}\n"
                f"Expected:\n"
                f"  - width2.torchparams (supernet weights)\n"
                f"  - width2.2.256.256.256.3.3.3.400.bn.tar (subnet config)\n"
                f"Found:\n"
                f"  - width2.torchparams: {supernet_path.exists()}\n"
                f"  - subnet config: {subnet_path.exists()}"
            )

        # Load model using local files
        # NOTE: Do NOT extract the .tar file - load it directly
        self.encoder = self._load_local_model(supernet_path, subnet_path)

        # Move to specified device
        self.encoder = self.encoder.to(device)

        # Set to eval mode first (required for BatchNorm with batch_size=1)
        self.encoder.eval()

        # Detect actual output dimension by running a dummy forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, 16000).to(device)
            dummy_output = self.encoder(dummy_input)
            self.output_dim = dummy_output.shape[-1]

        # Freeze parameters (critical for P-PhonMatchNet)
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("✓ All parameters frozen")

        # Display model information
        total_params = sum(p.numel() for p in self.encoder.parameters())
        trainable_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)

        print(f"✓ EfficientTDNN Small loaded successfully")
        print(f"  - Total parameters: {total_params:,} (≈ 899K)")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Output dimension: {self.output_dim}")
        print(f"  - Device: {self.device}")

    def _load_local_model(self, supernet_path, subnet_path):
        """
        Load EfficientTDNN model from local files

        This method manually constructs the model and loads weights,
        bypassing the HuggingFace dependency in WrappedModel.from_pretrained
        """
        # Load supernet weights
        sup_state_dict = torch.load(supernet_path, map_location='cpu', weights_only=False)['state_dict']

        # Load subnet configuration and BN statistics
        sub_state_dict = torch.load(subnet_path, map_location='cpu', weights_only=False)
        subnet_config = sub_state_dict['subnet']

        # Import sugar models
        from sugar.models import SpeakerModel
        from sugar.transforms import LogMelFbanks

        # Create transform (LogMelFbank with 80 channels)
        transform = LogMelFbanks(n_mels=80)

        # Create model architecture
        # EfficientTDNN Small uses tdnn8m2g with specific configuration
        # Output dimension: 192 (as per the actual pretrained weights)
        modelarch = tdnn8m2g(in_feats=80, out_embeds=192)

        # Wrap in SpeakerModel
        model = SpeakerModel(modelarch, transform=transform)

        # Wrap again (for consistent interface)
        from sugar.models import WrappedModel
        model = WrappedModel(model)

        # Load supernet weights
        model.load_state_dict(sup_state_dict, strict=False)

        # Clone subnet with specific configuration
        subnet = model.module.__S__.clone(subnet_config)

        # Load BN statistics
        subnet.load_state_dict(sub_state_dict['bn'], strict=False)

        # Wrap subnet in SpeakerModel with transform
        subnet_model = WrappedModel(SpeakerModel(subnet, transform=transform))

        return subnet_model

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embeddings from raw waveform

        Args:
            waveform: [B, T] @ 16kHz, raw audio waveform
                - B: batch size
                - T: number of time samples (e.g., 16000 for 1 second)

        Returns:
            embedding: [B, 400] speaker embedding

        Note:
            - No preprocessing required - model handles it internally
            - Uses torch.no_grad() when frozen (default)
            - Input must be 16kHz sampling rate
        """
        # Ensure waveform is on correct device
        if waveform.device != torch.device(self.device):
            waveform = waveform.to(self.device)

        # Extract embedding
        # Use no_grad when frozen (default), enable_grad when training
        with torch.no_grad() if not self.training else torch.enable_grad():
            embedding = self.encoder(waveform)

        # Verify output shape matches what we detected during init
        assert embedding.shape[-1] == self.output_dim, \
            f"Expected output dimension {self.output_dim}, got {embedding.shape[-1]}"

        return embedding  # [B, output_dim]

    def extract_from_audio_path(self, audio_path: str) -> torch.Tensor:
        """
        Extract speaker embedding from audio file path (convenience function)

        Args:
            audio_path: Path to audio file

        Returns:
            embedding: [1, 400] speaker embedding

        Note:
            - Automatically resamples to 16kHz if needed
            - Converts to mono if stereo
        """
        import torchaudio

        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Resample to 16kHz if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Remove channel dimension [1, T] -> [T]
        waveform = waveform.squeeze(0)

        # Add batch dimension [T] -> [1, T]
        waveform = waveform.unsqueeze(0)

        # Extract embedding
        return self.forward(waveform)

    def compute_similarity(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        metric: str = 'cosine'
    ) -> torch.Tensor:
        """
        Compute similarity between two speaker embeddings

        Args:
            embedding1: [B, 400] or [400] speaker embedding
            embedding2: [B, 400] or [400] speaker embedding
            metric: Similarity metric ('cosine' or 'euclidean')

        Returns:
            similarity: [B] or scalar similarity score
                - For cosine: range [-1, 1], higher is more similar
                - For euclidean: range [0, inf], lower is more similar
        """
        if metric == 'cosine':
            # Cosine similarity
            emb1_norm = nn.functional.normalize(embedding1, p=2, dim=-1)
            emb2_norm = nn.functional.normalize(embedding2, p=2, dim=-1)
            similarity = torch.sum(emb1_norm * emb2_norm, dim=-1)
            return similarity
        elif metric == 'euclidean':
            # Euclidean distance
            distance = torch.norm(embedding1 - embedding2, p=2, dim=-1)
            return distance
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'cosine' or 'euclidean'")

    @property
    def num_parameters(self) -> int:
        """Return total number of parameters"""
        return sum(p.numel() for p in self.encoder.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Return number of trainable parameters"""
        return sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)

    def load_finetuned_weights(self, checkpoint_path: str):
        """
        Load finetuned weights from checkpoint.
        
        Args:
            checkpoint_path: Path to finetuned checkpoint (.pt file)
                Expected format: {'model_state_dict': ..., 'eer': ..., 'embedding_dim': ...}
        
        Note:
            After loading, parameters remain frozen by default.
            The checkpoint is generated by speaker/finetune_speaker_encoder.py
        """
        import os
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Load state dict
        self.encoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # Move to device
        self.encoder = self.encoder.to(self.device)
        
        # Log info
        eer = checkpoint.get('eer', 'N/A')
        epoch = checkpoint.get('epoch', 'N/A')
        print(f"✓ Loaded finetuned speaker encoder")
        print(f"  - Checkpoint: {checkpoint_path}")
        print(f"  - Epoch: {epoch}")
        print(f"  - EER: {eer}%")
        
        # Re-freeze parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("Testing SpeakerEncoder")
    print("=" * 80)

    # Initialize encoder
    encoder = SpeakerEncoder(
        model_path='model/speaker/efficient_tdnn',
        freeze=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("\n" + "=" * 80)
    print("Running Tests")
    print("=" * 80)

    # Test 1: Output shape
    print("\n[Test 1] Output Shape")
    batch_size = 4
    duration = 1.0  # 1 second
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    waveform = torch.randn(batch_size, int(16000 * duration)).to(device)
    embedding = encoder(waveform)

    print(f"  Input shape: {waveform.shape}")
    print(f"  Output shape: {embedding.shape}")
    assert embedding.shape == (batch_size, 400), \
        f"Expected [4, 400], got {embedding.shape}"
    print("  ✓ PASSED: Output shape is [4, 400]")

    # Test 2: Parameters frozen
    print("\n[Test 2] Parameters Frozen")
    frozen_count = 0
    total_count = 0
    for param in encoder.parameters():
        total_count += 1
        if not param.requires_grad:
            frozen_count += 1

    print(f"  Total parameters: {total_count}")
    print(f"  Frozen parameters: {frozen_count}")
    assert frozen_count == total_count, "Not all parameters are frozen"
    print("  ✓ PASSED: All parameters are frozen")

    # Test 3: Deterministic output
    print("\n[Test 3] Deterministic Output")
    emb1 = encoder(waveform)
    emb2 = encoder(waveform)

    max_diff = torch.abs(emb1 - emb2).max().item()
    print(f"  Max difference: {max_diff}")
    assert torch.allclose(emb1, emb2, atol=1e-6), \
        "Same input should produce identical embeddings"
    print("  ✓ PASSED: Output is deterministic")

    # Test 4: Similarity computation
    print("\n[Test 4] Similarity Computation")
    emb_a = torch.randn(1, 400).to(device)
    emb_b = torch.randn(1, 400).to(device)

    cos_sim = encoder.compute_similarity(emb_a, emb_b, metric='cosine')
    euc_dist = encoder.compute_similarity(emb_a, emb_b, metric='euclidean')

    print(f"  Cosine similarity: {cos_sim.item():.4f}")
    print(f"  Euclidean distance: {euc_dist.item():.4f}")
    assert -1 <= cos_sim.item() <= 1, "Cosine similarity should be in [-1, 1]"
    print("  ✓ PASSED: Similarity computation works")

    # Test 5: Parameter count
    print("\n[Test 5] Parameter Count")
    num_params = encoder.num_parameters
    print(f"  Total parameters: {num_params:,}")
    print(f"  Expected: ~899,200")
    assert 850_000 <= num_params <= 950_000, \
        f"Parameter count {num_params} not in expected range [850K, 950K]"
    print("  ✓ PASSED: Parameter count is correct")

    print("\n" + "=" * 80)
    print("All Tests PASSED! ✓")
    print("=" * 80)
    print(f"\nSpeakerEncoder Summary:")
    print(f"  - Output dimension: 400")
    print(f"  - Total parameters: {encoder.num_parameters:,}")
    print(f"  - Trainable parameters: {encoder.num_trainable_parameters:,}")
    print(f"  - Device: {encoder.device}")
    print(f"  - Status: Ready for P-PhonMatchNet integration")
