"""
ColorizationUNet — replacement for ColorizationNet.

Same interface (input: L of shape (B,1,H,W); output: ab of shape (B,2,H,W))
but with U-Net-style skip connections for much sharper colorization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# =============================================================================
# Helper block: two consecutive conv layers with BatchNorm + ReLU.
# This is the "standard" U-Net decoder building block.
# - Two convs let the network learn richer transformations than one conv would.
# - BatchNorm stabilizes training (especially important with skip connections,
#   because concatenated features can have very different scales).
# - ReLU adds nonlinearity so the decoder can learn more than just a linear map.
# =============================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# =============================================================================
# Decoder "up" block:
#   1. Upsample the deep features (small spatial size, lots of channels).
#   2. Concatenate the matching encoder features (from a skip connection).
#   3. Run DoubleConv to fuse the two streams.
#
# The "skip" features come from a shallower part of the encoder, where spatial
# resolution is high and features are low-level (edges, textures). The
# upsampled deep features carry semantic info ("this is sky"). Concatenating
# them lets the decoder use BOTH "where" (skip) and "what" (deep) to predict
# pixel-accurate colors.
# =============================================================================
class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        # We use bilinear upsample + conv (rather than ConvTranspose2d) because:
        #   - It avoids the "checkerboard artifact" problem of transposed convs
        #   - It has fewer parameters and trains more reliably
        #   - It is the standard choice in modern U-Net variants
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear",
                                    align_corners=False)
        # After upsampling, channels = in_ch. After concat with skip, channels
        # = in_ch + skip_ch. DoubleConv then reduces that to out_ch.
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.upsample(x)
        # Edge case: if input H/W isn't a power of 2, the encoder's
        # downsampled feature map might be 1 pixel off from the upsampled
        # decoder map. Pad/crop to match before concatenating.
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
        # Concatenate along the channel dimension (dim=1).
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# =============================================================================
# The full U-Net colorization model.
#
# Architecture (for a 256x256 input):
#
#   INPUT:        L  (B, 1, 256, 256)
#                 ↓ repeat to 3 channels for ResNet
#                 (B, 3, 256, 256)
#
#   ENCODER (pretrained ResNet-18, frozen BN running stats kept):
#     stem        (B,  64, 128, 128)   ← captured as skip_1
#     layer1      (B,  64,  64,  64)   ← captured as skip_2
#     layer2      (B, 128,  32,  32)   ← captured as skip_3
#     layer3      (B, 256,  16,  16)   ← captured as skip_4
#     layer4      (B, 512,   8,   8)   ← bottleneck (most compressed)
#
#   DECODER (with skip connections at each stage):
#     up_4: (512, 8, 8)   + skip_4 (256, 16, 16) → (256, 16, 16)
#     up_3: (256, 16, 16) + skip_3 (128, 32, 32) → (128, 32, 32)
#     up_2: (128, 32, 32) + skip_2 ( 64, 64, 64) → ( 64, 64, 64)
#     up_1: ( 64, 64, 64) + skip_1 ( 64, 128,128)→ ( 32, 128, 128)
#
#   FINAL UPSAMPLE + HEAD:
#     upsample 2x        → (32, 256, 256)
#     conv 32 → 2        → (B, 2, 256, 256)  = predicted ab
#     tanh               → squashes to [-1, 1] which matches your ab/128
#                          normalization range
#
# Total parameters: ~14M (vs ~11M for the old ColorizationNet)
# =============================================================================
class ColorizationUNet(nn.Module):
    def __init__(self, pretrained=True, freeze_encoder_epochs=0):
        """
        Args:
            pretrained: load ImageNet weights for the ResNet-18 encoder
            freeze_encoder_epochs: if > 0, training loop should freeze encoder
                                   for this many epochs.
        """
        super().__init__()
        self.freeze_encoder_epochs = freeze_encoder_epochs

        # --- Build encoder from a pretrained ResNet-18 ---
        # We don't use nn.Sequential(*list(resnet.children())[:-2]) like the
        # old code, because we need to grab intermediate outputs (skip
        # connections). Instead, we split ResNet-18 into named stages.
        weights = "IMAGENET1K_V1" if pretrained else None
        resnet = models.resnet18(weights=weights)

        # ResNet-18's stem: conv1 + bn1 + relu + maxpool
        # Input  : (B, 3, 256, 256)
        # After  : (B, 64, 64, 64)  — downsampled by 4 already
        # We split this so we can grab the post-relu feature (before maxpool)
        # which is at the higher resolution (B, 64, 128, 128). That gives us
        # one more skip-connection stage and a sharper final output.
        self.stem_conv = resnet.conv1   # 7x7 conv, stride 2 → (B,64,128,128)
        self.stem_bn = resnet.bn1
        self.stem_relu = resnet.relu
        self.stem_pool = resnet.maxpool # stride 2 → (B,64,64,64)

        # The four residual stages of ResNet-18.
        # Each "layer" is actually a pair of BasicBlocks.
        self.layer1 = resnet.layer1     # → (B,  64, 64, 64)   no downsample
        self.layer2 = resnet.layer2     # → (B, 128, 32, 32)   downsample 2x
        self.layer3 = resnet.layer3     # → (B, 256, 16, 16)   downsample 2x
        self.layer4 = resnet.layer4     # → (B, 512,  8,  8)   downsample 2x

        # --- Build the decoder ---
        # Each UpBlock takes (deep_features, skip_features) and outputs the
        # next decoder stage. Channel counts are chosen to roughly mirror
        # the encoder.
        self.up4 = UpBlock(in_ch=512, skip_ch=256, out_ch=256)
        self.up3 = UpBlock(in_ch=256, skip_ch=128, out_ch=128)
        self.up2 = UpBlock(in_ch=128, skip_ch=64,  out_ch=64)
        self.up1 = UpBlock(in_ch=64,  skip_ch=64,  out_ch=32)

        # Final 2x upsample to get back to full input resolution (256x256),
        # then a 1x1 conv head to predict the 2 ab channels.
        self.final_upsample = nn.Upsample(scale_factor=2, mode="bilinear",
                                           align_corners=False)
        # 1x1 conv: just a per-pixel linear projection from 32 features → 2.
        # No BN here because we want the raw output to flow through tanh.
        self.head = nn.Conv2d(32, 2, kernel_size=1)

        # tanh squashes the output to [-1, 1]. This matches your ab
        # normalization (ab / 128), so the model's natural output range
        # aligns with the target range. The optimizer doesn't have to learn
        # to keep outputs bounded — the architecture enforces it.
        self.output_activation = nn.Tanh()

    # -------------------------------------------------------------------------
    # Helper to freeze/unfreeze the encoder. Useful for the first 2-3 epochs:
    # the decoder is randomly initialized while the encoder has pretrained
    # ImageNet features. Freezing the encoder lets the decoder "catch up"
    # before you start updating the encoder, which prevents the early random
    # decoder gradients from corrupting the pretrained encoder weights.
    # -------------------------------------------------------------------------
    def set_encoder_trainable(self, trainable: bool):
        for p in self.stem_conv.parameters(): p.requires_grad = trainable
        for p in self.stem_bn.parameters():   p.requires_grad = trainable
        for p in self.layer1.parameters():    p.requires_grad = trainable
        for p in self.layer2.parameters():    p.requires_grad = trainable
        for p in self.layer3.parameters():    p.requires_grad = trainable
        for p in self.layer4.parameters():    p.requires_grad = trainable


    def forward(self, L):
        """
        Args:
            L: (B, 1, H, W) Lightness channel in [0, 1] (after dividing by 100)

        Returns:
            ab: (B, 2, H, W) predicted A,B channels in [-1, 1]
                (matches your ab/128 target normalization)
        """
        input_size = L.shape[-2:]  # save for final resize (handles non-256 inputs)

        # ResNet-18 expects 3 channels — repeat L across channel dim.
        # This works because for grayscale, R==G==B and L is essentially
        # luminance, so the pretrained ImageNet features still fire usefully.
        x = L.repeat(1, 3, 1, 1)                  # (B, 3, H, W)

        # --- Encoder forward (capture skip features at each stage) ---
        # Run the stem in pieces so we can grab the pre-pool, post-ReLU
        # features as a higher-resolution skip.
        x = self.stem_conv(x)                     # (B, 64, H/2, W/2)
        x = self.stem_bn(x)
        x = self.stem_relu(x)
        skip_1 = x                                # (B, 64, H/2, W/2)   ← skip
        x = self.stem_pool(x)                     # (B, 64, H/4, W/4)

        skip_2 = self.layer1(x)                   # (B, 64, H/4, W/4)   ← skip
        skip_3 = self.layer2(skip_2)              # (B, 128, H/8, W/8)  ← skip
        skip_4 = self.layer3(skip_3)              # (B, 256, H/16,W/16) ← skip
        bottleneck = self.layer4(skip_4)          # (B, 512, H/32,W/32)
        # ^^ "bottleneck" — most compressed; semantic features only.

        # --- Decoder forward (upsample + concat skip + DoubleConv) ---
        d = self.up4(bottleneck, skip_4)          # (B, 256, H/16, W/16)
        d = self.up3(d,          skip_3)          # (B, 128, H/8,  W/8)
        d = self.up2(d,          skip_2)          # (B,  64, H/4,  W/4)
        d = self.up1(d,          skip_1)          # (B,  32, H/2,  W/2)

        # --- Final upsample + head ---
        d = self.final_upsample(d)                # (B, 32, H, W)
        ab = self.head(d)                         # (B,  2, H, W)
        ab = self.output_activation(ab)           # squash to [-1, 1]

        # Belt-and-suspenders: if H/W weren't multiples of 32, the final
        # spatial size could be 1 pixel off. Resize to be exactly safe.
        if ab.shape[-2:] != input_size:
            ab = F.interpolate(ab, size=input_size, mode="bilinear",
                               align_corners=False)
        return ab


# Quick sanity test — run this once to confirm shapes and parameter count.
if __name__ == "__main__":
    model = ColorizationUNet(pretrained=True)
    L = torch.randn(2, 1, 256, 256)
    ab = model(L)
    print(f"Input  L  shape: {L.shape}")
    print(f"Output ab shape: {ab.shape}")
    print(f"Output range: [{ab.min().item():.3f}, {ab.max().item():.3f}]  (should be in [-1, 1])")
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {n_params:,}")
    print(f"Trainable params: {n_trainable:,}")

    # Test the freeze helper
    model.set_encoder_trainable(False)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"After freeze: {n_trainable:,} trainable params (should be much smaller)")
