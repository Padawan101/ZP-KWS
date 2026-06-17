import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)


class Encoder(nn.Module):
    """Base class for encoders"""
    
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, src, src_mask=None):
        """
        Args:
            src      : source of shape `(batch, src_len)`
            src_mask : mask indicating the lengths of each source of shape `(batch, time)`
        """
        raise NotImplementedError


class AudioEncoder(Encoder):
    """Base class for audio encoders"""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.cnn = []
        self.rnn = []
        self.strides = [1]
        self.stride = 1
        last_input_features = kwargs['input_dim']
        bidirectional = kwargs.get('bidirectional', False)
        if kwargs['audio_input'] == 'raw':
            for l in kwargs['conv']:
                o, k, s, p = l
                self.cnn.append(nn.Conv1d(last_input_features, o, k, stride=s, padding=p)) # Torch does not support 'same' padding for stride > 1
                self.cnn.append(nn.BatchNorm1d(o))
                self.cnn.append(nn.ReLU())
                self.stride *= s
                self.strides.append(self.stride)
                last_input_features = o

            for l in kwargs['gru']:
                unit = l
                self.rnn.append(nn.GRU(last_input_features, unit[0], batch_first=True, bidirectional=bidirectional))
                last_input_features = unit[0] * 2 if bidirectional else unit[0]

        self.cnn = nn.ModuleList(self.cnn)
        self.rnn = nn.ModuleList(self.rnn)

        self.dense = nn.Linear(last_input_features, kwargs['fc'])
        self.act = nn.LeakyReLU()

    
    def forward(self, src, src_mask=None, verbose=False):
        """
        Args:
            src         : source of shape `(batch, time, feature)`
            src_mask    : mask indicating the lengths of each source of shape `(batch, time)`
        """
        # keep the batch mask
        if src_mask is not None:
            mask = src_mask[:,::self.stride]
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
        else:
            mask = None
        
        # [B, T, F]
        # cnn
        x = src.transpose(1, 2)
        for i, layer in enumerate(self.cnn): # [B, F, T] -> [B, Conv1d, T/self.stride]
            x = layer(x)
        x = x.transpose(1, 2)
        
        # rnn (with pack_padded_sequence for correct bidirectional support)
        if mask is not None and len(self.rnn) > 0:
            # Compute actual lengths from mask [B, T/stride, 1]
            lengths = mask.squeeze(-1).sum(dim=1).long().cpu()  # [B]
            lengths = lengths.clamp(min=1)  # avoid 0-length sequences
            x = x * mask  # zero out padding before packing
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            for layer in self.rnn:
                packed, _ = layer(packed)
            x, _ = pad_packed_sequence(packed, batch_first=True, total_length=mask.shape[1])
        else:
            hidden = None
            for layer in self.rnn:
                x, hidden = layer(x, hidden)
        
        x = self.dense(x)      # [B, T/self.stride, Dense]
        
        LD = x
        x = self.act(x)
        
        x = torch.nan_to_num(x) * mask
        mask = mask.squeeze(-1)

        return x, LD, mask


class EfficientAudioEncoder(Encoder):
    """Efficient encoder class for audio encoders"""
    
    def __init__(self, downsample=True, **kwargs):
        super().__init__()
        self.downsample = downsample
        self.layer = [] 
        self.deConv = None
        last_input_features = kwargs['input_dim']

        if self.downsample:
            self.layer.append(nn.Conv1d(last_input_features, kwargs['fc'], 5, stride=2, padding=2))
            self.layer.append(nn.BatchNorm1d(kwargs['fc']))
            self.layer.append(nn.ReLU())
            self.layer.append(nn.MaxPool1d(2))
            self.layer.append(nn.Conv1d(kwargs['fc'], kwargs['fc'], 5, stride=2, padding=2))
            self.layer.append(nn.BatchNorm1d(kwargs['fc']))
            self.layer.append(nn.ReLU())
            self.dense = nn.Linear(96, kwargs['fc'])
        else:
            self.layer.append(nn.Conv1d(last_input_features, kwargs['fc'], 3, stride=2, padding=1))
            self.layer.append(nn.BatchNorm1d(kwargs['fc']))
            self.layer.append(nn.ReLU())
            self.layer.append(nn.Conv1d(kwargs['fc'], kwargs['fc'], 3, stride=1, padding=1))
            self.layer.append(nn.BatchNorm1d(kwargs['fc']))
            self.layer.append(nn.ReLU())
            self.deConv = nn.ConvTranspose1d(96, kwargs['fc'], 5, 4) # 96: MAGIC NUM defined by google speech embedding
        self.layer = nn.Sequential(*self.layer)
        
        self.act = nn.LeakyReLU()

    def forward(self, src, src_mask=None, verbose=False, return_separate=False):
        """
        Args:
            src         : (spectrogram, gembed) where
                        : spectrogram - log mel-spectrogram of shape `(batch, time, mel)`
                        : gembed      - google speech embedding of shape `(batch, time / 8, 96)`
            src_mask    : mask indicating the lengths of spectrogram of shape `(batch, time)`
            return_separate : 是否返回分離的 conv_feat 和 gembed_processed
        
        Returns:
            如果 return_separate=False (預設):
                x: conv_feat + gembed_processed [B, T, 128]
                LD: conv_feat [B, T, 128]
                mask: [B, T]
            
            如果 return_separate=True:
                x: conv_feat + gembed_processed [B, T, 128]
                LD: conv_feat [B, T, 128]
                gembed_processed: [B, T, 128]  ← 新增
                mask: [B, T]
        """        
        spectrogram, gembed = src

        if src_mask is not None:
            s_mask, g_mask = src_mask
            if self.downsample:
                mask = s_mask[:,::8]
            else:
                mask = s_mask[:,::2]
            gembed = torch.nan_to_num(gembed) * g_mask.unsqueeze(-1)
        else:
            mask = None

        x = spectrogram.transpose(1, 2)
        x = self.layer(x)
        x = x.transpose(1, 2)

        LD = x  # conv_feat (純 conv 特徵)

        # [B, T/8, dense] or [B, T/2, dense]
        if self.downsample:
            y = self.act(self.dense(gembed))  # gembed_processed
            y_padded = nn.functional.pad(y, (0, 0, 0, x.shape[1] - y.shape[1], 0, 0), value=0.0)
            # Summation two embedding
            x = x + y_padded
        else:
            y = gembed.transpose(1, 2)
            y = self.act(self.deConv(y))
            y = y.transpose(1, 2)

            if x.shape[1] > y.shape[1]:
                y_padded = nn.functional.pad(y, (0, 0, 0, x.shape[1] - y.shape[1], 0, 0), value=0.0)
                x = x + y_padded
            elif x.shape[1] < y.shape[1]:
                y_padded = y[:, :x.shape[1], :]
                x = x + y_padded
            else:
                y_padded = y
                x = x + y
        
        x = torch.nan_to_num(x) * mask.unsqueeze(-1)
        LD = torch.nan_to_num(LD) * mask.unsqueeze(-1)

        # 新增：返回分離的特徵
        if return_separate:
            gembed_processed = torch.nan_to_num(y_padded) * mask.unsqueeze(-1)
            return x, LD, gembed_processed, mask

        return x, LD, mask


class HybridAudioEncoder(Encoder):
    """
    Hybrid Audio Encoder: AudioEncoder's powerful conv+GRU+dense for raw audio
    + EfficientAudioEncoder's DeConv for gembed fusion.
    
    Combines the best of both worlds:
    - Raw audio branch: deep conv + GRU for strong temporal modeling (→ better LDN)
    - Gembed branch: DeConv projection for frozen pretrained features
    - Fusion: simple addition at T/stride resolution
    
    Returns same format as EfficientAudioEncoder: (emb_s, LDN, mask)
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        
        # ===== Raw Audio Branch (from AudioEncoder) =====
        self.cnn = nn.ModuleList()
        self.rnn = nn.ModuleList()
        self.strides = [1]
        self.stride = 1
        last_input_features = kwargs['input_dim']
        
        for l in kwargs['conv']:
            o, k, s, p = l
            self.cnn.append(nn.Conv1d(last_input_features, o, k, stride=s, padding=p))
            self.cnn.append(nn.BatchNorm1d(o))
            self.cnn.append(nn.ReLU())
            self.stride *= s
            self.strides.append(self.stride)
            last_input_features = o
        
        bidirectional = kwargs.get('bidirectional', False)
        for l in kwargs['gru']:
            unit = l
            self.rnn.append(nn.GRU(last_input_features, unit[0], batch_first=True, bidirectional=bidirectional))
            last_input_features = unit[0] * 2 if bidirectional else unit[0]
        
        self.dense = nn.Linear(last_input_features, kwargs['fc'])
        self.act = nn.LeakyReLU()
        
        # ===== Gembed Branch (from EfficientAudioEncoder) =====
        # DeConvTranspose: 96 → fc, kernel=5, stride=4  (upsamples T/8 → ~T/2)
        self.deConv = nn.ConvTranspose1d(96, kwargs['fc'], 5, 4)
        self.act_gembed = nn.LeakyReLU()
        
        print(f">> [HybridAudioEncoder]")
        print(f">>   Raw branch: {len(kwargs['conv'])} conv layers + {len(kwargs['gru'])} GRU layers (Bidirectional: {bidirectional}) + Dense")
        print(f">>   Conv config: {kwargs['conv']}")
        print(f">>   GRU config: {kwargs['gru']}")
        print(f">>   Total stride: {self.stride}")
        print(f">>   Gembed branch: DeConvTranspose(96→{kwargs['fc']}, k=5, s=4)")
    
    def forward(self, src, src_mask=None, verbose=False, return_separate=False):
        """
        Args:
            src: (spectrogram, gembed)
                spectrogram: [B, T, mel]
                gembed: [B, T/8, 96]
            src_mask: (s_mask, g_mask)
            return_separate: if True, also return gembed_processed separately
        
        Returns:
            emb_s: [B, T/stride, fc]  fused features (LDN + gembed)
            LDN: [B, T/stride, fc]    raw audio features only (pre-activation)
            mask: [B, T/stride]        output mask
            (optional) gembed_processed: [B, T/stride, fc]
        """
        spectrogram, gembed = src
        
        # === Mask ===
        if src_mask is not None:
            s_mask, g_mask = src_mask
            mask = s_mask[:, ::self.stride]  # downsample mask by total stride
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)  # [B, T/stride, 1] for broadcasting
            gembed = torch.nan_to_num(gembed) * g_mask.unsqueeze(-1)
        else:
            mask = None
        
        # === Raw Audio Branch (Conv + GRU + Dense) ===
        x = spectrogram.transpose(1, 2)  # [B, mel, T]
        for layer in self.cnn:
            x = layer(x)                 # [B, fc, T/stride]
        x = x.transpose(1, 2)           # [B, T/stride, fc]
        
        # GRU (with pack_padded_sequence for correct bidirectional support)
        if mask is not None and len(self.rnn) > 0:
            # Compute actual lengths from mask [B, T/stride, 1]
            lengths = mask.squeeze(-1).sum(dim=1).long().cpu()  # [B]
            lengths = lengths.clamp(min=1)  # avoid 0-length sequences
            x = x * mask  # zero out padding before packing
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            for layer in self.rnn:
                packed, _ = layer(packed)
            x, _ = pad_packed_sequence(packed, batch_first=True, total_length=mask.shape[1])
        else:
            hidden = None
            for layer in self.rnn:
                x, hidden = layer(x, hidden)
        
        # Dense (matching AudioEncoder: LD = pre-activation, x = post-activation)
        x = self.dense(x)    # [B, T/stride, fc]
        LDN = x              # pre-activation = Latent Discriminative features
        x = self.act(x)      # post-activation for fusion
        
        # === Gembed Branch (DeConv) ===
        y = gembed.transpose(1, 2)           # [B, 96, T/8]
        y = self.act_gembed(self.deConv(y))  # [B, fc, ~T/2]
        y = y.transpose(1, 2)               # [B, ~T/2, fc]
        
        # Align lengths (DeConv output may differ by ±1 frame)
        if x.shape[1] > y.shape[1]:
            y = nn.functional.pad(y, (0, 0, 0, x.shape[1] - y.shape[1], 0, 0), value=0.0)
        elif x.shape[1] < y.shape[1]:
            y = y[:, :x.shape[1], :]
        
        # === Fusion ===
        emb_s = x + y
        
        # Apply mask
        if mask is not None:
            emb_s = torch.nan_to_num(emb_s) * mask
            LDN = torch.nan_to_num(LDN) * mask
        
        # Squeeze mask back to 2D [B, T/stride]
        if mask is not None and mask.dim() == 3:
            mask = mask.squeeze(-1)
        
        if return_separate:
            gembed_processed = torch.nan_to_num(y) * mask.unsqueeze(-1) if mask is not None else y
            return emb_s, LDN, gembed_processed, mask
        
        return emb_s, LDN, mask


class EnhancedGembedEncoder(Encoder):
    """
    Enhanced Gembed Encoder: DeConv upsampling + GRU temporal processing.
    
    與 gemb-only (AudioEncoder on gembed) 的差異：
    1. 使用 DeConv 上採樣到 T/2 而非 Conv 下採樣到 T/16
    2. GRU 在 T/2 解析度工作，能看到 fine-grained phoneme boundaries
    3. 單一 stream output，無加法融合的稀釋問題
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        
        fc = kwargs['fc']  # 128
        bidirectional = kwargs.get('bidirectional', False)
        
        # === DeConv Upsampling (T/8 → ~T/2) ===
        self.deConv = nn.ConvTranspose1d(96, fc, 5, 4)
        self.bn_deconv = nn.BatchNorm1d(fc)
        self.act_deconv = nn.LeakyReLU()
        
        # === GRU (optional, skip if gru_layers=0) ===
        self.rnn = nn.ModuleList()
        last_input_features = fc
        for l in kwargs['gru']:
            unit = l
            self.rnn.append(nn.GRU(
                last_input_features, unit[0], 
                batch_first=True, bidirectional=bidirectional
            ))
            last_input_features = unit[0] * 2 if bidirectional else unit[0]
        
        # === Dense (only if GRU exists) ===
        self.has_gru = len(self.rnn) > 0
        if self.has_gru:
            self.dense = nn.Linear(last_input_features, fc)
            self.act = nn.LeakyReLU()
        
        # === Stride info ===
        self.stride = 2
        
        mode = "DeConv + BN + LeakyReLU"
        if self.has_gru:
            mode += f" + GRU×{len(self.rnn)}(bi={bidirectional}) + Dense"
        print(f">> [EnhancedGembedEncoder] {mode}")
        print(f">>   Output resolution: T/2, dim={fc}")
    
    def forward(self, src, src_mask=None, verbose=False, return_separate=False):
        """
        Args:
            src: (spectrogram, gembed)
                spectrogram: [B, T, mel] — 此 encoder 不使用，但為保持介面一致而接收
                gembed: [B, T/8, 96]
            src_mask: (s_mask, g_mask)
            
        Returns:
            emb_s: [B, T/2, fc]
            LDN: [B, T/2, fc]  — pre-activation (for aux CE)
            mask: [B, T/2]
        """
        spectrogram, gembed = src
        
        # === Mask ===
        if src_mask is not None:
            s_mask, g_mask = src_mask
            mask = s_mask[:, ::2]
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            gembed = torch.nan_to_num(gembed) * g_mask.unsqueeze(-1)
        else:
            mask = None
        
        # === DeConv Upsampling (T/8 → ~T/2) ===
        y = gembed.transpose(1, 2)                              # [B, 96, T/8]
        y = self.act_deconv(self.bn_deconv(self.deConv(y)))     # [B, fc, ~T/2]
        y = y.transpose(1, 2)                                   # [B, ~T/2, fc]
        
        # === Align with mask length ===
        target_len = mask.shape[1] if mask is not None else y.shape[1]
        if y.shape[1] > target_len:
            y = y[:, :target_len, :]
        elif y.shape[1] < target_len:
            y = nn.functional.pad(y, (0, 0, 0, target_len - y.shape[1], 0, 0), value=0.0)
        
        if self.has_gru:
            # === GRU (with pack_padded_sequence) ===
            if mask is not None and len(self.rnn) > 0:
                lengths = mask.squeeze(-1).sum(dim=1).long().cpu()
                lengths = lengths.clamp(min=1)
                y = y * mask
                packed = pack_padded_sequence(y, lengths, batch_first=True, enforce_sorted=False)
                for layer in self.rnn:
                    packed, _ = layer(packed)
                y, _ = pad_packed_sequence(packed, batch_first=True, total_length=mask.shape[1])
            else:
                hidden = None
                for layer in self.rnn:
                    y, hidden = layer(y, hidden)
            
            # === Dense ===
            y = self.dense(y)
            LDN = y                 # pre-activation → aux_head
            emb_s = self.act(y)    # post-activation → Self-Attention
        else:
            # === DeConv-only mode (gru_layers=0) ===
            # DeConv output (已過 BN + LeakyReLU) 直接作為 emb_s 和 LDN
            LDN = y
            emb_s = y
        
        # === Apply mask ===
        if mask is not None:
            emb_s = torch.nan_to_num(emb_s) * mask
            LDN = torch.nan_to_num(LDN) * mask
        
        if mask is not None and mask.dim() == 3:
            mask = mask.squeeze(-1)
        
        if return_separate:
            gembed_processed = emb_s
            return emb_s, LDN, gembed_processed, mask
        
        return emb_s, LDN, mask


class TextEncoder(Encoder):
    """Base class for text encoders"""
    
    def __init__(self, **kwargs):
        super().__init__()
        
        self.features = kwargs['text_input']
        self.vocab = kwargs['vocab']
        if self.features == 'phoneme':
            self.dense = nn.Linear(kwargs['vocab'], kwargs['fc'])
        elif self.features == 'g2p_embed':
            self.dense = nn.Linear(256, kwargs['fc'])
        self.act = nn.LeakyReLU()

    def forward(self, src, verbose=False):
        """
        Args:
            src         : phoneme token of shape `(batch, phoneme, *)`
                        : [WARNING] for 'g2p_embed' features, shape is `(batch, phoneme, 256)`
            src_mask    : mask indicating the lengths of each source of shape `(batch, time)`
        """
        # [B, phoneme] -> [B, phoneme, embedding]
        x = src
        src_mask = (src != 0.0)
        if src_mask.dim() == 3:
            src_mask = src_mask[:,:,0]
        
        if self.features == 'phoneme':
            x = nn.functional.one_hot(x.to(torch.int64), num_classes=self.vocab).to(torch.float)
        x = self.act(self.dense(x))
        x = torch.nan_to_num(x) * src_mask.unsqueeze(-1)

        return x, src_mask