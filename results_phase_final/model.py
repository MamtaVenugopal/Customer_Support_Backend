"""
Autoencoder backbones for bearing anomaly detection.

Every model has the same signature:

    forward(x) -> (recon, z)

where
    x     : (B, 1, H, W) log-mel
    recon : reconstruction of x (shape may differ by padding; train/eval
            align the time axis to min length before computing loss)
    z     : (B, C, h, w) encoder bottleneck feature map; pooled to
            (B, C) for the domain classifier / contrastive head

Every model exposes `embed_dim` = C so the domain head can be built
with the correct input size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# 1) SimpleAE — the original tiny baseline (fast, low capacity)
# --------------------------------------------------------------------------
class SimpleAE(nn.Module):
    embed_dim = 32

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 2, stride=2),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z


# --------------------------------------------------------------------------
# 2) UNetAE — encoder/decoder with skip connections
# --------------------------------------------------------------------------
def _conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
    )


def _dec_block(in_c, out_c):
    """Decoder conv block used by U-Net style decoders."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
    )


class UNetAE(nn.Module):
    """
    2-level U-Net with bottleneck. Pads the time axis to the next
    multiple of 4 so skip shapes align, then crops back at the end.
    """
    embed_dim = 128

    def __init__(self):
        super().__init__()
        self.enc1 = _conv_block(1, 32)
        self.enc2 = _conv_block(32, 64)
        self.bottleneck = _conv_block(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = _conv_block(128, 64)  # 64 (up2) + 64 (enc2)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = _conv_block(64, 32)   # 32 (up1) + 32 (enc1)

        self.out_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        # pad H and W to multiples of 4
        pad_h = (4 - H % 4) % 4
        pad_w = (4 - W % 4) % 4
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        e1 = self.enc1(x)                    # (B, 32, H', W')
        e2 = self.enc2(F.max_pool2d(e1, 2))  # (B, 64, H'/2, W'/2)
        z  = self.bottleneck(F.max_pool2d(e2, 2))  # (B,128, H'/4, W'/4)

        d2 = self.up2(z)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        out = self.out_conv(d1)

        # crop back to original (H, W)
        out = out[..., :H, :W]
        return out, z


# --------------------------------------------------------------------------
# 3) MobileNetAE — ImageNet-pretrained MobileNetV2 encoder + conv-transpose decoder
# --------------------------------------------------------------------------
class MobileNetAE(nn.Module):
    """
    Encoder = MobileNetV2 features[0:14] (output: 96 channels @ 1/16).
    Decoder = 4x transposed-conv upsampling back to the input resolution.
    Single-channel log-mel is repeated to 3 channels so the ImageNet
    weights can be used unchanged.
    """
    embed_dim = 96

    def __init__(self, pretrained=True):
        super().__init__()
        try:
            from torchvision.models import mobilenet_v2
            if pretrained:
                from torchvision.models import MobileNet_V2_Weights
                bb = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            else:
                bb = mobilenet_v2(weights=None)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "MobileNetAE requires torchvision. Install with `pip install torchvision`."
            ) from e

        # features[0..13] : output (96, H/16, W/16)
        self.encoder = nn.Sequential(*list(bb.features.children())[:14])

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16,  1, 4, stride=2, padding=1),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x3 = x.repeat(1, 3, 1, 1)           # (B, 3, H', W')
        z = self.encoder(x3)                # (B, 96, H'/16, W'/16)
        out = self.decoder(z)               # (B, 1, H', W')
        out = out[..., :H, :W]
        return out, z


# --------------------------------------------------------------------------
# 4) UNetMobileNetEncoderAE — pretrained MobileNetV2 encoder + U-Net decoder
# --------------------------------------------------------------------------
class UNetMobileNetEncoderAE(nn.Module):
    """
    Encoder:
      - MobileNetV2 features split into 4 stages with skips
      - input log-mel (1ch) is repeated to 3ch for ImageNet weights

    Decoder:
      - U-Net style upsampling with skip concatenations from encoder stages
      - final crop restores original (H, W)
    """
    embed_dim = 96

    def __init__(self, pretrained=True):
        super().__init__()
        try:
            from torchvision.models import mobilenet_v2
            if pretrained:
                from torchvision.models import MobileNet_V2_Weights
                bb = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            else:
                bb = mobilenet_v2(weights=None)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "UNetMobileNetEncoderAE requires torchvision. Install with `pip install torchvision`."
            ) from e

        feats = list(bb.features.children())
        # Spatial scale progression (roughly):
        # e0: /2, e1: /4, e2: /8, z: /16
        self.enc0 = nn.Sequential(*feats[:2])    # ~16 ch
        self.enc1 = nn.Sequential(*feats[2:4])   # ~24 ch
        self.enc2 = nn.Sequential(*feats[4:7])   # ~32 ch
        self.enc3 = nn.Sequential(*feats[7:14])  # ~96 ch (bottleneck)

        self.up2 = nn.ConvTranspose2d(96, 64, 2, stride=2)
        self.dec2 = _dec_block(64 + 32, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = _dec_block(32 + 24, 32)
        self.up0 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec0 = _dec_block(16 + 16, 16)
        self.up_in = nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.out_conv = nn.Conv2d(16, 1, 1)

    @staticmethod
    def _resize_like(x, ref):
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        B, C, H, W = x.shape
        # MobileNet downsampling path reaches /16 -> pad to multiple of 16
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x3 = x.repeat(1, 3, 1, 1)
        e0 = self.enc0(x3)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        z = self.enc3(e2)

        u2 = self._resize_like(self.up2(z), e2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self._resize_like(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        u0 = self._resize_like(self.up0(d1), e0)
        d0 = self.dec0(torch.cat([u0, e0], dim=1))
        u_in = self._resize_like(self.up_in(d0), x)
        out = self.out_conv(u_in)

        out = out[..., :H, :W]
        return out, z


# --------------------------------------------------------------------------
# Heads (shared across backbones, sized by each backbone's embed_dim)
# --------------------------------------------------------------------------
class DomainClassifier(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self, z):
        z = z.mean(dim=[2, 3])
        return self.fc(z)


def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    sim = torch.matmul(z1, z2.T)
    labels = torch.arange(z1.size(0)).to(z1.device)
    return F.cross_entropy(sim / temperature, labels)


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------
ARCHES = {
    "ae": SimpleAE,
    "unet": UNetAE,
    "mobilenet": MobileNetAE,
    "unet_mobilenet_encoder": UNetMobileNetEncoderAE,
}


def build_model(arch: str) -> nn.Module:
    if arch not in ARCHES:
        raise ValueError(
            f"Unknown arch '{arch}'. Available: {list(ARCHES.keys())}"
        )
    return ARCHES[arch]()


# backward-compat alias for any old import like `from model import AE`
AE = SimpleAE
