import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models, transforms
from PIL import Image
import numpy as np

from skimage.color import rgb2lab, lab2rgb  # for LAB↔RGB conversion


# =========================
# 1. Model definition
# =========================

class ColorizationNet(nn.Module):
    """
    Colorization network that:
    - Uses a pretrained ResNet-18 as encoder (on ImageNet)
    - Adds a small decoder to predict A,B color channels in LAB space
    """
    def __init__(self):
        super().__init__()

        # Load a pretrained ResNet-18 (ImageNet weights)[web:48]
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Take only the convolutional "feature extractor" part (remove avgpool + fc)
        # children() returns the layers in order;[:-2] keeps everything up to the last conv block
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        # Decoder: upsample feature maps back to original size and output 2 channels (A and B)
        # This is a very simple decoder; you can make it deeper for better quality.
        self.decoder = nn.Sequential(
            # input: (batch, 512, H/32, W/32) for ResNet-18
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),
            # no activation: we predict raw A,B in a normalized range
        )

    def forward(self, L):
        """
        Forward pass.

        Args:
            L: Tensor of shape (batch, 1, H, W) with Lightness channel normalized to [0,1]

        Returns:
            ab: Tensor of shape (batch, 2, H, W) with predicted A,B channels
        """
        # ResNet expects 3 channels; duplicate L to fake 3-channel input
        x = L.repeat(1, 3, 1, 1)  # shape: (batch, 3, H, W)

        # Extract deep features with encoder
        feats = self.encoder(x)   # shape: (batch, 512, H/32, W/32)

        # Decode features into A,B maps
        ab_small = self.decoder(feats)  # shape: (batch, 2, H/2, W/2 or H,W depending on upsampling)

        # Resize decoder output to exactly match input spatial size (H, W)
        ab = F.interpolate(ab_small, size=L.shape[2:], mode='bilinear', align_corners=False)

        return ab

