# local_colorization_dataset.py
import os
from glob import glob

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.color import rgb2lab  # RGB↔LAB conversion[web:48][web:51]


class LocalColorizationDataset(Dataset):
    """
    Loads RGB images from a local folder and returns:
      L_tensor: (1, H, W) in [0,1]
      ab_tensor: (2, H, W) in approx [-1,1]
    """

    def __init__(self, root_dir, image_size=256, extensions=("jpg", "jpeg", "png")):
        """
        Args:
            root_dir: folder containing images (optionally subfolders).
            image_size: resize all images to (image_size, image_size) for training.
            extensions: file extensions to include.
        """
        super().__init__()
        self.root_dir = root_dir
        self.image_size = image_size

        # Collect all image paths recursively
        self.image_paths = []
        print(f"Scanning {root_dir} for images...")
        for ext in extensions:
            self.image_paths.extend(
                glob(os.path.join(root_dir, "**", f"*.{ext}"), recursive=True)
            )
        self.image_paths = sorted(self.image_paths)
        print(f"Found {len(self.image_paths)} images.")

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {root_dir}")

        # Resize transform
        self.resize = transforms.Resize((image_size, image_size))
        print(f"Dataset initialized with {len(self.image_paths)} images. Ready to load and process.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns:
            L_tensor: torch.FloatTensor (1,H,W), Lightness in [0,1]
            ab_tensor: torch.FloatTensor (2,H,W), A,B normalized to ~[-1,1]
        """
        path = self.image_paths[idx]

        # Load RGB image
        img = Image.open(path).convert("RGB")

        # Resize to fixed training size
        img = self.resize(img)

        # To numpy in [0,1]
        img_np = np.asarray(img) / 255.0  # (H,W,3)

        # RGB → LAB
        lab = rgb2lab(img_np).astype("float32")  # (H,W,3)[web:48]

        # Split channels
        L = lab[..., 0] / 100.0        # [0,1]
        ab = lab[..., 1:] / 128.0      # roughly [-1,1]

        # To torch tensors
        L_tensor = torch.from_numpy(L).unsqueeze(0)                  # (1,H,W)
        ab_tensor = torch.from_numpy(np.transpose(ab, (2, 0, 1)))    # (2,H,W)

        return L_tensor, ab_tensor
