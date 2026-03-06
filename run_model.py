import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models, transforms
from PIL import Image
import numpy as np

from skimage.color import rgb2lab, lab2rgb  # for LAB↔RGB conversion

from class_ColorizationNet import ColorizationNet

import os



# =========================
# Preprocessing helpers
# =========================

def load_grayscale_image_as_L(path):
    """
    Load an image from disk, convert to LAB, and return:

    - L: Lightness channel as torch tensor (1, 1, H, W), normalized to [0,1]
    - L_np: same L as numpy, for reconstruction
    - H, W: spatial size of the original image

    No resizing is done, so output will match input dimensions.
    """
    # Load image as RGB
    img = Image.open(path).convert("RGB")

    # Convert to numpy in [0,1]
    img_np = np.asarray(img) / 255.0  # shape: (H,W,3)

    # RGB → LAB
    lab = rgb2lab(img_np).astype("float32")

    # L channel in [0,1]
    L = lab[..., 0] / 100.0          # (H,W)
    L_np = L.copy()

    # Torch tensor (1,1,H,W)
    L_tensor = torch.from_numpy(L).unsqueeze(0).unsqueeze(0)

    H, W = L.shape
    return L_tensor, L_np, H, W



def lab_Lab_to_rgb(L_np, ab_np):
    """
    Take L (numpy, H,W, in [0,1] scaled from [0,100]) and A,B (numpy, H,W, scaled)
    and convert back to RGB numpy image in [0,1].

    Args:
        L_np: numpy array of shape (H,W), L in [0,1]
        ab_np: numpy array of shape (2,H,W) or (H,W,2), A,B normalized

    Returns:
        rgb: numpy array of shape (H,W,3) in [0,1]
    """
    # Ensure AB has shape (H,W,2)
    if ab_np.ndim == 3 and ab_np.shape[0] == 2:
        # (2,H,W) → (H,W,2)
        ab_np = np.transpose(ab_np, (1, 2, 0))

    # Denormalize:
    # - L: back to [0,100]
    # - A,B: assume model outputs in [-1,1]; rescale to roughly [-128,128]
    L_channel = L_np * 100.0
    ab_channel = ab_np * 128.0

    # Stack into LAB image: shape (H,W,3)
    lab = np.zeros((L_np.shape[0], L_np.shape[1], 3), dtype="float32")
    lab[..., 0] = L_channel
    lab[..., 1:] = ab_channel

    # Convert LAB → RGB in [0,1][web:51]
    rgb = lab2rgb(lab)
    return rgb


# =========================
# Inference: gray → color
# =========================

def colorize_image(model, image_path, device="cpu"):
    """
    Full pipeline:
    - Load gray image from disk
    - Run through model to get A,B
    - Reconstruct color RGB image

    Args:
        model: ColorizationNet instance with (ideally) trained weights
        image_path: path to grayscale (or RGB) input image
        device: "cpu" or "cuda"

    Returns:
        rgb_out: numpy array (H,W,3) in [0,1] with colorized image
    """
    model.eval()                      # inference mode (no dropout/bn updates)
    model.to(device)

    # Prepare L channel input
    L_tensor, L_np, H, W = load_grayscale_image_as_L(image_path)
    L_tensor = L_tensor.to(device).float()  # ensure float32

    with torch.no_grad():             # no gradients for inference
        # Forward pass: predict A,B channels
        ab_pred = model(L_tensor)     # tensor (1,2,H,W)

    # Move predicted A,B to CPU and convert to numpy
    ab_pred_np = ab_pred.cpu().numpy()[0]  # shape (2,H,W)

    # Reconstruct RGB image from L and A,B
    rgb_out = lab_Lab_to_rgb(L_np, ab_pred_np)  # (H,W,3) in [0,1]

    return rgb_out





# =========================
# Example usage
# =========================

if __name__ == "__main__":
    # Create model instance with default parameters
    model = ColorizationNet()

    # TODO: if you have a trained checkpoint, load it here:

    # ckpt_path = "checkpoints_local\\Landscape Dataset 90K_colorization_step_1000.pt"  # pick latest/best
    ckpt_path = "checkpoints_s3\\Landscape Dataset 90K_best.pt"  # pick latest/best
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])


    # Path to input grayscale image (can also be color; we only use its L channel)
    # input_path = "input_gray_or_rgb.jpg"
    input_path = "test_images/old lady.jpg"

    # Run colorization
    rgb_colorized = colorize_image(model, input_path, device="cpu")

    # Convert result to uint8 [0,255] and save with PIL
    rgb_uint8 = (np.clip(rgb_colorized, 0, 1) * 255).astype("uint8")
    out_img = Image.fromarray(rgb_uint8)
    # out_img.save("output_colorized.png")
    os.makedirs("model_outputs", exist_ok=True)
    out_img.save(f"model_outputs/{os.path.splitext(os.path.basename(input_path))[0]}_colorized.png")