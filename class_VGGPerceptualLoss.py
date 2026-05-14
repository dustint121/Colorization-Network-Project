# =========================
# Perceptual loss (VGG-based)
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_Weights


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG-16 features pretrained on ImageNet.

    Takes two RGB images (predicted and ground truth), runs each through
    frozen VGG-16, and compares features at multiple layers using L1.

    Inputs are expected in [0, 1] range.
    """
    def __init__(self, layer_indices=(3, 8, 15, 22), device="cuda"):
        super().__init__()
        # Indices into vgg16.features:
        #   3  → after relu1_2  (low-level: edges, color)
        #   8  → after relu2_2  (textures)
        #   15 → after relu3_3  (parts)
        #   22 → after relu4_3  (objects)
        # Mixing layers gives multi-scale perceptual feedback.
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        vgg = vgg.to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False

        self.vgg = vgg
        self.layer_indices = set(layer_indices)
        self.max_layer = max(layer_indices)

        # ImageNet normalization — VGG expects this.
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _extract_features(self, x):
        """Run x through VGG and return a list of feature maps at the chosen layers."""
        x = (x - self.mean) / self.std
        feats = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layer_indices:
                feats.append(x)
            if i >= self.max_layer:
                break
        return feats

    def forward(self, pred_rgb, target_rgb):
        """
        pred_rgb, target_rgb: (B, 3, H, W) in [0, 1].
        Returns scalar perceptual loss.
        """
        pred_feats = self._extract_features(pred_rgb)
        with torch.no_grad():
            target_feats = self._extract_features(target_rgb)

        loss = 0.0
        for pf, tf in zip(pred_feats, target_feats):
            loss = loss + F.l1_loss(pf, tf)
        return loss / len(pred_feats)


def lab_to_rgb_torch(L, ab):
    """
    Convert LAB tensors back to RGB on the same device.
    L:  (B, 1, H, W) in [0, 1]   (i.e. raw L / 100)
    ab: (B, 2, H, W) in [-1, 1]  (i.e. raw ab / 128)
    Returns RGB in [0, 1].

    Uses kornia for differentiable conversion (must be differentiable so
    gradients flow back to the model's ab predictions).
    """
    from kornia.color import lab_to_rgb
    lab = torch.cat([L * 100.0, ab * 128.0], dim=1)  # un-normalize
    rgb = lab_to_rgb(lab)
    return rgb.clamp(0.0, 1.0)