# Speaker Encoder Module

EfficientTDNN Small speaker encoder for P-PhonMatchNet.

## Overview

This module provides a frozen speaker encoder based on EfficientTDNN Small for extracting 400-dimensional speaker embeddings from raw audio waveforms.

**Model Specifications:**
- Architecture: EfficientTDNN Small (2 layers, [256, 256, 256] channels, [3, 3, 3] kernels)
- Parameters: 899.20K (all frozen)
- Output: 400-dimensional speaker embedding
- Input: Raw waveform @ 16kHz

## Files

- `encoder.py` - Main SpeakerEncoder class
- `efficient_tdnn/` - Pre-trained model weights
  - `width2.torchparams` - Supernet weights (34MB)
  - `width2.2.256.256.256.3.3.3.400.bn.tar` - Subnet configuration (80KB)
- `sugar/` - Sugar library (source code, not pip package)

## Quick Start

### Using Docker (Recommended)

The easiest way to test the speaker encoder is using Docker:

```bash
# From project root
./run_speaker_test.sh
```

This will:
1. Build Docker image with all dependencies
2. Run tests to verify output shape is [B, 400]
3. Check parameter freezing
4. Test determinism and similarity computation

### Manual Testing

If you have all dependencies installed:

```bash
python3 test_speaker_encoder_simple.py
```

### Dependencies

Required packages (included in Docker):
- PyTorch >= 2.2.1
- scipy
- thop
- geatpy
- torchaudio
- huggingface_hub

## Usage Example

```python
from model.speaker.encoder import SpeakerEncoder
import torch

# Initialize encoder (frozen by default)
encoder = SpeakerEncoder(
    model_path='model/speaker/efficient_tdnn',
    freeze=True,
    device='cuda'
)

# Extract embeddings from waveform
waveform = torch.randn(4, 16000).cuda()  # 4 samples, 1 second @ 16kHz
embedding = encoder(waveform)  # [4, 400]

# Compute speaker similarity
cos_sim = encoder.compute_similarity(embedding[0], embedding[1], metric='cosine')
print(f"Similarity: {cos_sim.item():.4f}")
```

## Integration with P-PhonMatchNet

The SpeakerEncoder is designed to be integrated into P-PhonMatchNet v1.3:

```python
# In P-PhonMatchNet forward pass
from model.speaker.encoder import SpeakerEncoder

# Initialize (done in __init__)
self.speaker_encoder = SpeakerEncoder(freeze=True, device='cuda')

# Use in forward (Path B - Speaker Verification)
with torch.no_grad():
    # Get enrollment embedding
    enrollment_emb = self.speaker_encoder(enrollment_audio)  # [B, 400]

    # Get input audio embedding
    input_emb = self.speaker_encoder(input_audio)  # [B, 400]

    # Compute similarity
    P_spk = self.speaker_encoder.compute_similarity(
        input_emb, enrollment_emb, metric='cosine'
    )
    # Map to [0, 1]: P_spk = (cosine + 1) / 2
    P_spk = (P_spk + 1.0) / 2.0
```

## Important Notes

1. **Output Dimension**: The output is **400-d**, not 512-d. This is specific to EfficientTDNN Small.

2. **Frozen Parameters**: All parameters are frozen by default (`requires_grad=False`). This is intentional - the speaker encoder is only used for feature extraction.

3. **16kHz Input**: The model expects raw waveforms sampled at 16kHz. No preprocessing needed - the model handles it internally.

4. **No HuggingFace Download**: The model loads from local files in `efficient_tdnn/`. No internet connection required after initial setup.

5. **Sugar Library**: The sugar library is included as source code in `sugar/`. Import path handling is done automatically in `encoder.py`.

## Verification

Expected test results:

```
✅ All Tests PASSED!

SpeakerEncoder Summary:
  - Model: EfficientTDNN Small
  - Output dimension: 400
  - Total parameters: ~899,200
  - Trainable parameters: 0
  - Device: cuda/cpu
  - Status: ✅ Ready for P-PhonMatchNet integration
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError` for sugar modules:
- Use Docker environment (recommended)
- Or manually install: `pip install scipy thop geatpy`

### CUDA Out of Memory

The speaker encoder uses ~900K parameters and should not cause memory issues. If you encounter OOM:
- Ensure `freeze=True` (default)
- Use `torch.no_grad()` during inference
- Reduce batch size

### Wrong Output Dimension

If output is not [B, 400]:
- Check you're using the correct subnet config file
- Verify the subnet config is `(2, [256, 256, 256], [3, 3, 3], 400)`

## References

- Paper: Wang et al., "EfficientTDNN: Efficient Architecture Search for Speaker Recognition", IEEE/ACM TASLP, 2022
- arXiv: https://arxiv.org/abs/2103.13581
- HuggingFace: https://huggingface.co/mechanicalsea/efficient-tdnn
- GitHub: https://github.com/mechanicalsea/sugar
