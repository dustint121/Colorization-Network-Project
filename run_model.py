import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models, transforms
from PIL import Image
import numpy as np

from skimage.color import rgb2lab, lab2rgb  # for LAB↔RGB conversion

from class_ColorizationUNet import ColorizationUNet

import os



# =========================
# Preprocessing helpers
# =========================

def load_image_as_L(path, image_size=None):
    """
    Load an image from disk, convert to LAB, and return two L representations:

    - L_model:  (1, 1, image_size, image_size) torch tensor, normalized to
                [0, 1]. This is what the network sees. Always at the
                training resolution so activations stay in-distribution.

    - L_full:   (H_orig, W_orig) numpy array, normalized to [0, 1]. This
                is the *original*-resolution L channel, used to compose
                the final output image so the visible luminance detail
                (edges, textures) is preserved at native resolution.

    - (H_orig, W_orig): original spatial size of the source image.

    """
    # Load and normalize once at the original resolution.
    img_full = Image.open(path).convert("RGB")
    W_orig, H_orig = img_full.size  # PIL uses (width, height)

    # ---- Full-resolution L (for final composition) ----
    img_full_np = np.asarray(img_full, dtype=np.float32) / 255.0
    lab_full = rgb2lab(img_full_np).astype(np.float32)
    L_full = lab_full[..., 0] / 100.0        # (H_orig, W_orig) in [0, 1]

    # ---- Model-resolution L (for the network forward pass) ----
    if image_size is not None:
        img_model = img_full.resize((image_size, image_size), Image.BILINEAR)
    else:
        img_model = img_full
    img_model_np = np.asarray(img_model, dtype=np.float32) / 255.0
    lab_model = rgb2lab(img_model_np).astype(np.float32)
    L_model_np = lab_model[..., 0] / 100.0   # (image_size, image_size)

    L_model = torch.from_numpy(L_model_np).unsqueeze(0).unsqueeze(0)  # (1,1,s,s)

    return L_model, L_full, (H_orig, W_orig)



def compose_lab_to_rgb(L_np, ab_np):
    """
    Merge an L channel and an ab pair into an RGB image.

    Args:
        L_np:  numpy (H, W)   with L in [0, 1]  (rescaled internally to [0, 100])
        ab_np: numpy (2, H, W) or (H, W, 2) with ab normalized to roughly [-1, 1]
               (rescaled internally to roughly [-128, 128])

        L_np and ab_np MUST already be at the same (H, W). Any resizing
        of ab is the caller's responsibility (see resize_ab_to).

    Returns:
        rgb: numpy (H, W, 3) in [0, 1]
    """
    if ab_np.ndim == 3 and ab_np.shape[0] == 2:
        ab_np = np.transpose(ab_np, (1, 2, 0))          # (H, W, 2)

    assert L_np.shape == ab_np.shape[:2], (
        f"L/ab spatial size mismatch: L={L_np.shape}, ab={ab_np.shape[:2]}"
    )

    lab = np.zeros((L_np.shape[0], L_np.shape[1], 3), dtype=np.float32)
    lab[..., 0]  = L_np * 100.0
    lab[..., 1:] = ab_np * 128.0

    return lab2rgb(lab)



def resize_ab_to(ab_np, target_hw):
    """
    Bilinearly resize a normalized ab tensor to a target spatial size.

    Args:
        ab_np:     numpy (2, H_src, W_src), ab in the model's normalized range
        target_hw: (H_dst, W_dst) tuple

    Returns:
        numpy (2, H_dst, W_dst).

    Note: bilinear is the right interpolation here - ab is a chroma
    signal (smooth almost everywhere; sharp changes only at object
    boundaries, which we intentionally soften slightly to hide
    quantization noise from the model). This is exactly how JPEG
    reconstructs its 4:2:0-subsampled chroma planes at display time.
    """
    H_dst, W_dst = target_hw
    # Torch bilinear upsample is the same math skimage/JPEG use, plus
    # keeps things vectorized on CPU without another dependency.
    ab_t = torch.from_numpy(ab_np).unsqueeze(0)  # (1, 2, H_src, W_src)
    ab_up = F.interpolate(
        ab_t, size=(H_dst, W_dst), mode="bilinear", align_corners=False
    )
    return ab_up.squeeze(0).numpy()               # (2, H_dst, W_dst)



# =========================
# Inference: gray → color
# =========================

def colorize_image(model, image_path, device="cpu", image_size=256,
                   output_size="original"):
    """
    Full pipeline:

    1. Load the source image.
    2. Build a resized L for the network AND keep a full-res L for
       compositing.
    3. Run the model on the resized L to get a 256x256 ab prediction.
    4. Upsample ab back to the original HxW (or to image_size).
    5. Merge with the chosen L to produce the final RGB.

    Args:
        model:       ColorizationUNet with (ideally) trained weights.
        image_path:  path to grayscale or color input image.
        device:      "cpu" or "cuda".
        image_size:  training resolution used for the network forward
                     pass (default 256; must match S3ColorizationDataset).
        output_size: "original" (default) → final image matches the
                     input resolution; "model" → final image is
                     image_size x image_size (old behavior, useful for
                     tensor-space comparisons and debugging).

    Returns:
        rgb_out:     numpy (H_out, W_out, 3) in [0, 1] - the colorized image
        ab_pred_np:  numpy (2, image_size, image_size) - RAW model output
                     (before any upsampling), returned for diagnostics
    """
    if output_size not in ("original", "model"):
        raise ValueError(
            f"output_size must be 'original' or 'model', got {output_size!r}"
        )

    model.eval()
    model.to(device)

    # Prepare both L representations.
    L_model, L_full, (H_orig, W_orig) = load_image_as_L(
        image_path, image_size=image_size
    )
    L_model = L_model.to(device).float()

    with torch.no_grad():
        ab_pred = model(L_model)                 # (1, 2, image_size, image_size)

    ab_pred_np = ab_pred.cpu().numpy()[0]        # (2, image_size, image_size)

    if output_size == "original":
        # Upsample the smooth ab prediction to the source resolution,
        # then compose with the full-resolution (SHARP) L channel.
        # This yields a native-resolution output whose luminance detail
        # matches the input exactly and whose chroma comes from the
        # trained model.
        ab_up = resize_ab_to(ab_pred_np, (H_orig, W_orig))
        rgb_out = compose_lab_to_rgb(L_full, ab_up)
    else:
        # Old behavior: everything at model resolution (image_size²).
        L_model_np = L_model.squeeze().cpu().numpy()   # (image_size, image_size)
        rgb_out = compose_lab_to_rgb(L_model_np, ab_pred_np)

    return rgb_out, ab_pred_np




if __name__ == "__main__":
    # IMAGE_SIZE = the resolution the network was TRAINED at.
    IMAGE_SIZE = 256

    # OUTPUT_SIZE = "original" → save at the input image's native HxW
    #             = "model"    → save at IMAGE_SIZE x IMAGE_SIZE 
    OUTPUT_SIZE = "original"


    # Create model instance with default parameters
    model = ColorizationUNet()

    # ckpt_path = "checkpoints_local\\landscape-images_colorization_best.pt"
    # ckpt_path = "checkpoints_local\\imdb-images_colorization_best.pt"
    # ckpt_path = "checkpoints_local\\same-image_colorization_best.pt"
    # ckpt_path = "checkpoints_local\\imagenet21k_a100_highram_best_old.pt"
    # ckpt_path = "checkpoints_local\\imagenet21k_a100_highram_best.pt"
    ckpt_path = "checkpoints_local\\ilsvrc-image-net_colorization_best.pt"

    #get substring of ckpt_path between 'checkpoints_local\\' and '_best.pt'
    ckpt_name = ckpt_path.split("checkpoints_local\\")[1].split("_best.pt")[0]

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])

    # Path to input grayscale image (can also be color; we only use its L channel)
    # input_path = "test_images/old lady.jpg"
    # input_path = "test_images/lady.jpg"
    # input_path = "test_images/michael-jordan.jpg"
    # input_path = "test_images/president_obama.jpg"
    # input_path = "test_images/test.jpg"
    # input_path = "test_images/group.jpg"
    # input_path = "test_images/landscape_bw.jpg"
    # input_path = "test_images/fantasy.jpg"
    input_path = "test_images/person.jpg"
    # input_path = "test_images/ronald_reagan.jpg"



    rgb_colorized, ab_pred_np = colorize_image(
        model, input_path,
        device="cpu",
        image_size=IMAGE_SIZE,
        output_size=OUTPUT_SIZE,
    )

    rgb_uint8 = (np.clip(rgb_colorized, 0, 1) * 255).astype("uint8")
    out_img = Image.fromarray(rgb_uint8)
    os.makedirs(f"model_outputs/{ckpt_name}", exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_path))[0]
    out_path = f"model_outputs/{ckpt_name}/{stem}_colorized.png"
    out_img.save(out_path)
    print(f"wrote colorized image ({out_img.size[0]}x{out_img.size[1]}) to: "
          f"{out_path}")

