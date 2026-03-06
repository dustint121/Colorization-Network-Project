# class_S3ColorizationDataset.py
import io
import os
import torch
from torch.utils.data import IterableDataset
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.color import rgb2lab

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
        """List all image files under S3 prefix."""
        # print("Listing S3 keys using boto3...")
        # print(f"\tBucket: {self.bucket}")
        # print(f"\tPrefix: {self.prefix}")
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
        # print(f"\tTotal image keys found: {len(keys)}")
        return keys

    def _reader_to_sample(self, reader_or_key):
        """Convert S3 reader/key to (L, ab) sample."""
        if hasattr(reader_or_key, "read"):
            # s3torchconnector S3Reader
            buffer = reader_or_key.read()
        else:
            # boto3 key string - CREATE NEW CLIENT HERE (per sample, picklable)
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_KEY,
                endpoint_url=self.endpoint,
                region_name=self.region,
            )
            buffer = s3_client.get_object(
                Bucket=self.bucket, Key=reader_or_key
            )["Body"].read()
            # print(f"Read {len(buffer)} bytes from S3 for key: {reader_or_key}")

        img = Image.open(io.BytesIO(buffer)).convert("RGB")
        img = self.resize(img)
        img_np = np.asarray(img) / 255.0

        lab = rgb2lab(img_np).astype("float32")
        L = lab[..., 0] / 100.0
        ab = lab[..., 1:] / 128.0

        L_tensor = torch.from_numpy(L).unsqueeze(0)
        ab_tensor = torch.from_numpy(np.transpose(ab, (2, 0, 1)))

        return L_tensor, ab_tensor

    def __iter__(self):
        if self.use_s3torchconnector:
            # s3torchconnector path
            for reader in self.dataset:
                try:
                    yield self._reader_to_sample(reader)
                except Exception:
                    continue
        else:
            # boto3 fallback
            for key in self.keys:
                try:
                    yield self._reader_to_sample(key)
                except Exception:
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