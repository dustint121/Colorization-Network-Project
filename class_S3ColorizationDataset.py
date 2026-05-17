# class_S3ColorizationDataset.py
import io
import os
import torch
from torch.utils.data import IterableDataset
from torchvision import transforms
import torch.nn.functional as F
from torchvision.io import decode_image
from kornia.color import rgb_to_lab

import dotenv
dotenv.load_dotenv()

S3_AVAILABLE = False
try:
    import s3torchconnector as s3tc
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    # print("s3torchconnector not available, will use boto3 fallback")

import boto3

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")


class S3ColorizationDataset(IterableDataset):
    """
    Iterable dataset for S3. Falls back to boto3 if s3torchconnector unavailable (Windows).

    NOTE: All preprocessing is CPU-only. DataLoader workers are forked subprocesses,
    and CUDA does not survive fork() reliably — touching CUDA inside a worker
    silently corrupts samples (or raises async errors that get swallowed by the
    try/except in __iter__). We move tensors to GPU in the training loop instead,
    via L_batch.to(device) / ab_batch.to(device).
    """
    def __init__(
        self,
        s3_prefix_or_uris,
        region,
        endpoint,
        image_size=256,
        split="train",  # "train" or "val"
        val_fraction=0.1,
        use_s3torchconnector=True,
    ):
        super().__init__()
        self.image_size = image_size
        self.region = region
        self.endpoint = endpoint
        self.split = split
        self.resize = transforms.Resize((image_size, image_size))
        self.num_files = None

        # Windows check: s3torchconnector doesn't work on Windows
        self.use_s3torchconnector = use_s3torchconnector and S3_AVAILABLE and os.name != "nt"

        if self.use_s3torchconnector:
            self._build_s3torchconnector_dataset(s3_prefix_or_uris)
        else:
            self._build_boto3_dataset(s3_prefix_or_uris)

    def _build_s3torchconnector_dataset(self, prefix_or_uris):
        """Use s3torchconnector (Linux/Mac)."""
        self.bucket = prefix_or_uris.split("/")[2]  if isinstance(prefix_or_uris, str) else None  # s3://bucket/prefix → bucket
        self.prefix = "/".join(prefix_or_uris.split("/")[3:])  if isinstance(prefix_or_uris, str) else None  # rest of path

        if isinstance(prefix_or_uris, str):
            self.dataset = s3tc.S3IterableDataset.from_prefix(
                prefix_or_uris,
                region=self.region,
                endpoint=self.endpoint,
                enable_sharding=True  # Enable sharding for distributed training
            )
            self.num_files = len(self._list_s3_keys())
        else:
            self.dataset = s3tc.S3IterableDataset.from_objects(
                prefix_or_uris,
                region=self.region,
                endpoint=self.endpoint,
                enable_sharding=True  # Enable sharding for distributed training
            )
            self.num_files = len(prefix_or_uris)
        # print(f"Size={len(self.dataset)}")

    def _build_boto3_dataset(self, prefix_or_uris):
        """Fallback for Windows using boto3."""
        # NO CLIENT HERE - will create per sample in _reader_to_sample
        if isinstance(prefix_or_uris, str):
            self.bucket = prefix_or_uris.split("/")[2]  if isinstance(prefix_or_uris, str) else None  # s3://bucket/prefix → bucket
            self.prefix = "/".join(prefix_or_uris.split("/")[3:])  if isinstance(prefix_or_uris, str) else None  # rest of path
            self.keys = self._list_s3_keys() if self.bucket and self.prefix else []
            print(f"Extracted bucket: {self.bucket}, prefix: {self.prefix} from URI: {prefix_or_uris if isinstance(prefix_or_uris, str) else prefix_or_uris[0]}")

        if isinstance(prefix_or_uris, list):
            # If a list of URIs is provided, extract bucket and prefix from the first URI
            first_uri = prefix_or_uris[0] #"s3://landscape-images/00146.jpg"
            self.bucket = first_uri.split("/")[2]  # s3://bucket/prefix → bucket
            # from first_url extract prefix by removing "s3://bucket/" from the start of the URI
            self.prefix = "/".join(first_uri.split("/")[3:])  # rest of path
            #remove the extensions from each URI in the list to get the keys
            self.keys = [uri.split(f"s3://{self.bucket}/")[1] for uri in prefix_or_uris]  # Extract keys from URIs
            # print(f"Extracted bucket: {self.bucket}, prefix: {self.prefix} from first URI: {first_uri}")

        # print(f"Found {len(self.keys)} image keys in S3 with bucket: {self.bucket} and prefix: {self.prefix}")

    def _list_s3_keys(self):
        """List all image files under S3 prefix, with on-disk cache."""
        import pickle, hashlib

        # Unique cache file per (bucket, prefix) combo
        # cache_key = hashlib.md5(f"{self.bucket}/{self.prefix}".encode()).hexdigest()[:12]
        cache_key = f"{self.bucket}_{self.prefix}".replace("/", "_")[:50]  # simpler cache key
        cache_path = f"/content/s3_keys_cache_{cache_key}.pkl"
        if not os.path.exists("/content"):
            cache_path = f"./s3_keys_cache_{cache_key}.pkl"

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                keys = pickle.load(f)
            print(f"\tLoaded {len(keys)} keys from cache: {cache_path}")
            return keys

        print(f"\tListing S3 keys (no cache found)...")
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            endpoint_url=self.endpoint,
            region_name=self.region,
        )
        paginator = s3_client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.lower().endswith((".jpg", ".jpeg", ".png")):
                    keys.append(key)

        with open(cache_path, "wb") as f:
            pickle.dump(keys, f)
        print(f"\tCached {len(keys)} keys to: {cache_path}")
        return keys



    def __iter__(self):
        if self.use_s3torchconnector:
            # s3torchconnector path
            for reader in self.dataset:
                try:
                    yield self._reader_to_sample(reader)
                except Exception as e:
                    # Log instead of silently dropping — was hiding the CUDA-in-worker bug.
                    print(f"[S3ColorizationDataset] dropped sample: {type(e).__name__}: {e}")
                    continue
        else:
            # boto3 fallback
            for key in self.keys:
                try:
                    yield self._reader_to_sample(key)
                except Exception as e:
                    print(f"[S3ColorizationDataset] dropped sample {key!r}: {type(e).__name__}: {e}")
                    continue

    def __len__(self):
        if self.use_s3torchconnector:
            return self.num_files
        else:
            return len(self.keys)




    def s3_train_val_folders_exists(self, bucket_name):
        """
        Checks if 'train' and 'val' folders exist under the given S3 prefix by listing objects with that prefix.
        """

        s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY,
                                    endpoint_url=self.endpoint, region_name=self.region)

        if "s3://" in bucket_name:
            bucket_name = bucket_name.replace("s3://", "").split("/")[0]  # Extract bucket name from prefix

        # List a maximum of one object with the given prefix
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix="train/",  # Check for 'train' folder
            MaxKeys=1 # Only need one result to confirm existence
        )
        train_exists = response["KeyCount"] > 0
        if train_exists:
            print("Found 'train' folder in S3.")
        else:
            print("Did NOT find 'train' folder in S3.")
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix="val/",  # Check for 'val' folder
            MaxKeys=1 # Only need one result to confirm existence
        )
        val_exists = response["KeyCount"] > 0
        if val_exists:
            print("Found 'val' folder in S3.")
        else:
            print("Did NOT find 'val' folder in S3.")


        # If 'Contents' is in the response, it means at least one object exists
        # within that prefix, so the 'folder' exists.
        return train_exists and val_exists


    def _reader_to_sample(self, reader_or_key):
        """
        Read one S3 object → RGB tensor → LAB → (L, ab).

        CPU-only. DataLoader workers cannot safely use CUDA (fork-vs-CUDA issue),
        and we'd have to copy back to CPU anyway for the collate step. The main
        training loop moves batches to GPU via .to(device).
        """
        # --- 1) Read bytes from S3 or s3torchconnector ---
        if hasattr(reader_or_key, "read"):
            buffer = reader_or_key.read()
        else:
            s3_client = getattr(self, "_s3_client", None)
            if s3_client is None:
                s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=AWS_ACCESS_KEY,
                    aws_secret_access_key=AWS_SECRET_KEY,
                    endpoint_url=self.endpoint,
                    region_name=self.region,
                )
                self._s3_client = s3_client
            buffer = s3_client.get_object(
                Bucket=self.bucket, Key=reader_or_key
            )["Body"].read()

        # --- 2) Decode JPEG/PNG → tensor (C,H,W), uint8 ---
        byte_tensor = torch.as_tensor(memoryview(buffer), dtype=torch.uint8).clone()
        img_tensor = decode_image(byte_tensor)

        # Normalize to [0,1]
        img_tensor = img_tensor.float() / 255.0  # (C,H,W)

        # --- 3) CPU preprocessing (always) ---
        # Ensure 3 channels (grayscale → fake RGB by repeating)
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
        elif img_tensor.shape[0] == 4:
            # Drop alpha if present (RGBA → RGB)
            img_tensor = img_tensor[:3]

        # Resize on CPU
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )[0]                                          # (3,H',W')

        # kornia rgb_to_lab runs on whatever device the tensor is on — here, CPU.
        lab_tensor = rgb_to_lab(img_tensor.unsqueeze(0))[0]  # (3,H',W')
        L  = lab_tensor[0:1] / 100.0
        ab = lab_tensor[1:]  / 128.0

        if not hasattr(self, "_debug_done"):
            print("=== _reader_to_sample DEBUG ===")
            print("  use_s3torchconnector:", self.use_s3torchconnector)
            print("  CUDA available:", torch.cuda.is_available(), "(not used in workers)")
            print("  L shape:", L.shape, "ab shape:", ab.shape)
            print("  L dtype:", L.dtype, "ab dtype:", ab.dtype)
            print("  L min/max:", float(L.min()), float(L.max()))
            print("  ab min/max:", float(ab.min()), float(ab.max()))
            print("  any NaN in L:", torch.isnan(L).any().item(),
                  "any NaN in ab:", torch.isnan(ab).any().item())
            self._debug_done = True

        return L, ab