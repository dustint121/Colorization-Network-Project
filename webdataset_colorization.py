"""
Streams shuffled WebDataset tar shards from Wasabi/AWS S3 as (L, ab) tensor pairs.
Drop-in replacement for the per-object `S3ColorizationDataset` path.



Output contract (matches the old dataset exactly)
-------------------------------------------------
Each sample is `(L, ab)` where:
  - L  : float32 tensor, shape (1, H, W), values roughly in [0.0, 1.0]   (LAB-L / 100)
  - ab : float32 tensor, shape (2, H, W), values roughly in [-1.0, 1.0]  (LAB-ab / 128)

Same as `S3ColorizationDataset._reader_to_sample`, so the U-Net, the loss,
and the visualization code all work unchanged.

Shard layout (produced by reshard_imagenet.py)
----------------------------------------------
  s3://image-net-21k-shards/shard-00000.tar  ┐
  s3://image-net-21k-shards/shard-00001.tar  ├─  ~1000 JPEGs per shard,
  ...                                        │   globally shuffled across
  s3://image-net-21k-shards/shard-13128.tar  ┘   classes at reshard time.
  First TRAIN_FRACTION (95%) of shards = train, last 5% = val.
"""

from __future__ import annotations

import io
import os
import random
import warnings
from typing import Optional

import boto3
from botocore.config import Config
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

# Kornia's rgb_to_lab is ~5× faster than skimage's rgb2lab on CPU and matches
# to within ~0.5 ab units. Same one training already used.
from kornia.color import rgb_to_lab

import webdataset as wds
from webdataset.gopen import gopen_schemes


# ============================================================================
# Boto3 client cache (per-process — DataLoader workers each get their own)
# ============================================================================
# DataLoader workers fork the parent process. Boto3 sessions are NOT fork-safe:
# if we created the client in the parent, workers would share an underlying
# HTTP pool and race. So we lazily construct one client per worker process on
# first use.

_BOTO3_CLIENT = None
_BOTO3_CONFIG: dict = {}


def _get_s3_client():
    """Return this process's cached boto3 S3 client, creating it on first use."""
    global _BOTO3_CLIENT
    if _BOTO3_CLIENT is None:
        # Retries + timeouts tuned for streaming 100 MB tars from Wasabi:
        # - connect_timeout: fail fast on network hiccups (10s)
        # - read_timeout:    tars stream at ~40 MB/s, so 120s covers a full one
        # - max_pool_conns:  DataLoader workers each get ONE boto3 client with
        #                    a pool of up to 8 connections — plenty for
        #                    downloading one tar at a time
        cfg = Config(
            connect_timeout=10,
            read_timeout=120,
            retries={"max_attempts": 5, "mode": "adaptive"},
            max_pool_connections=8,
        )
        _BOTO3_CLIENT = boto3.client(
            "s3",
            aws_access_key_id=_BOTO3_CONFIG["access_key"],
            aws_secret_access_key=_BOTO3_CONFIG["secret_key"],
            endpoint_url=_BOTO3_CONFIG["endpoint"],
            region_name=_BOTO3_CONFIG["region"],
            config=cfg,
        )
    return _BOTO3_CLIENT


# ============================================================================
# Register an s3:// handler with WebDataset's gopen dispatch table
# ============================================================================
# WebDataset 1.x calls `gopen.gopen(url)` internally from `tarfile_to_samples`
# to open each shard URL. The dispatch table `gopen_schemes` maps URL schemes
# ("http", "gs", "ais", ...) to handler functions with signature
# `(url, mode, bufsize, **kw) -> file-like`. `s3://` is NOT registered by
# default, which is why our earlier attempt to intercept URLs via a pipeline
# stage failed with "no gopen handler defined": tarfile_to_samples ignored
# our upstream {stream: ...} and called gopen.gopen() on the raw URL anyway.
#
# The correct integration point is to register `s3` here. Any pipeline that
# uses `wds.tarfile_to_samples()` will then find our handler automatically.

def _gopen_s3(url: str, mode: str = "rb", bufsize: int = 8192, **kw):
    """Open `s3://bucket/key` for reading, returning a file-like BytesIO.

    We read the entire object into memory (BytesIO). Shards are ~100 MB each,
    which is trivial on any Colab/EC2 box, and tarfile prefers seekable
    streams -- boto3's StreamingBody isn't seekable, which makes tar reads
    much slower.

    Signature matches `gopen_schemes` contract: (url, mode, bufsize, **kw).
    """
    if mode != "rb":
        raise ValueError(f"s3 gopen only supports mode='rb', got {mode!r}")
    _, _, rest = url.partition("s3://")
    bucket, _, key = rest.partition("/")
    resp = _get_s3_client().get_object(Bucket=bucket, Key=key)
    return io.BytesIO(resp["Body"].read())


# Install the handler. Safe to re-register on notebook re-runs.
gopen_schemes["s3"] = _gopen_s3


# ============================================================================
# Per-sample decode: (JPEG bytes) → (L, ab) tensor pair
# ============================================================================
def _decode_sample(sample: dict, image_size: int):
    """Convert a WebDataset sample dict → (L, ab) float32 tensors.

    Mirrors S3ColorizationDataset._reader_to_sample:
      Image.open(...).convert("RGB").resize((s, s), BILINEAR)
      -> numpy in [0, 1]
      -> LAB
      -> L = lab[0] / 100, ab = lab[1:] / 128
    """
    # WebDataset stores each tar member under a key derived from its extension.
    # ImageNet-21K JPEGs are "n01440764_10026.JPEG" → extension is "JPEG".
    # WebDataset lowercases the extension, so the sample dict has "jpeg".
    img_bytes = (
        sample.get("jpeg")
        or sample.get("jpg")
        or sample.get("png")
    )
    if img_bytes is None:
        # Fallback: any bytes value (defensive; shouldn't hit in practice).
        for v in sample.values():
            if isinstance(v, (bytes, bytearray)):
                img_bytes = v
                break
        if img_bytes is None:
            raise ValueError(
                f"sample has no decodable image bytes: keys={list(sample.keys())}"
            )

    # PIL is universal — torchvision.io.decode_image needs libjpeg-turbo in
    # some builds, which fails on Colab occasionally. PIL always works.
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)

    # (H, W, 3) uint8 → (3, H, W) float32 in [0, 1]
    img_t = torch.from_numpy(
        __import__("numpy").asarray(img, dtype="float32")
    ) / 255.0
    img_t = img_t.permute(2, 0, 1).contiguous()          # (3, H, W)

    # RGB → LAB on CPU (kornia stays on CPU when input is on CPU)
    lab = rgb_to_lab(img_t.unsqueeze(0))[0]              # (3, H, W)
    L = lab[0:1] / 100.0                                  # (1, H, W) ≈ [0, 1]
    ab = lab[1:] / 128.0                                  # (2, H, W) ≈ [-1, 1]
    return L, ab


def _decode_sample_safe(sample: dict, image_size: int):
    """Decode with error swallowing.

    Corrupt JPEG in a shard normally kills the worker (uncaught in the map
    stage). Instead we skip: return None and filter downstream.
    """
    try:
        return _decode_sample(sample, image_size)
    except Exception as e:
        warnings.warn(
            f"[webdataset] skipping bad sample "
            f"key={sample.get('__key__', '?')}: {type(e).__name__}: {e}"
        )
        return None


# ============================================================================
# Shard listing (called once at startup)
# ============================================================================
def _list_shards(bucket: str) -> list[str]:
    """Return sorted list of shard keys (`shard-00000.tar`, ...) in `bucket`."""
    client = _get_s3_client()
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.endswith(".tar") and not k.startswith("_"):
                keys.append(k)
    keys.sort()  # deterministic order → deterministic train/val split
    return keys


# ============================================================================
# Pipeline builder
# ============================================================================
def make_webdataset(
    shard_urls: list[str],
    image_size: int = 256,
    shuffle_shards: bool = True,
    shuffle_buffer: int = 4000,
    seed: int = 42,
) -> wds.DataPipeline:
    """Build a WebDataset pipeline that yields (L, ab) sample pairs.

    Stages:
      1. SimpleShardList(shard_urls)         — the shard URLs
      2. shuffle(shard-level, optional)      — shard order per epoch
      3. split_by_node / split_by_worker     — one shard per (rank, worker)
      4. tarfile_to_samples                   — opens shards via gopen (our
                                                 registered `s3` handler above)
                                                 and yields grouped sample dicts
      5. shuffle(sample-level, optional)     — decorrelate samples within a shard
      6. map(_decode_sample_safe)             — JPEG → (L, ab)
      7. filter(not None)                     — drop failed decodes
    """
    stages: list = [wds.SimpleShardList(shard_urls, seed=seed)]

    if shuffle_shards:
        stages.append(
            wds.shuffle(
                len(shard_urls),
                initial=len(shard_urls),
                rng=random.Random(seed),
            )
        )

    stages += [
        wds.split_by_node,              # multi-GPU (DDP): rank-disjoint shards
        wds.split_by_worker,            # multi-worker: worker-disjoint shards
        # `tarfile_to_samples` calls `gopen.gopen()` internally; because we
        # registered `s3` in `gopen_schemes` above, s3:// URLs Just Work.
        wds.tarfile_to_samples(handler=wds.handlers.warn_and_continue),
    ]

    if shuffle_buffer > 1:
        stages.append(wds.shuffle(shuffle_buffer))

    stages += [
        wds.map(lambda s: _decode_sample_safe(s, image_size)),
        wds.select(lambda x: x is not None),
    ]

    return wds.DataPipeline(*stages)


# ============================================================================
# Public entry point: build train + val DataLoaders
# ============================================================================
def build_webdataset_loaders(
    s3_bucket: str,
    region: str,
    endpoint: str,
    *,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    batch_size: int = 32,
    image_size: int = 256,
    num_workers: int = 8,
    train_fraction: float = 0.95,
    shuffle_buffer: int = 4000,
    seed: int = 42,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    images_per_shard: int = 1000,
    val_num_workers: Optional[int] = None,
):
    """Return `(train_loader, val_loader)` over the shards in `s3_bucket`.

    Iterating each loader yields `(L_batch, ab_batch)` tensors of shape
    `(B, 1, H, W)` and `(B, 2, H, W)` — identical to the per-object dataset.

    Args:
        s3_bucket:      bucket name (no leading "s3://").
        region:         AWS-style region (e.g. "us-west-2"). Wasabi ignores
                        this but boto3 wants a non-empty value.
        endpoint:       full Wasabi endpoint URL.
        aws_access_key_id / aws_secret_access_key:
                        credentials. Falls back to env vars if None.
        batch_size:     per-loader batch size.
        image_size:     resize target used at decode time (must match the
                        `run_model.py` inference IMAGE_SIZE and the model's
                        pretrained scale — 256 for ResNet-18 U-Net).
        num_workers:    DataLoader workers per train loader.
        train_fraction: fraction of shards used for train (rest for val).
                        Split is at shard granularity — statistically fine
                        because shards were globally shuffled at reshard time.
        shuffle_buffer: sample-level shuffle buffer size for train.
                        Shards are already pre-shuffled globally, so a small
                        buffer (a few thousand) is enough to decorrelate
                        adjacent samples from the same shard.
        seed:           RNG seed for shard-shuffle and dataloader shuffling.
        persistent_workers / prefetch_factor: standard DataLoader knobs.
        images_per_shard: only used to estimate epoch length. Off-by-one on
                        the last shard is harmless.
        val_num_workers: default = max(2, num_workers // 4). Val is only ~5%
                        of data, so it doesn't benefit from as many workers.

    Notes:
        - Val loader has `shuffle_shards=False` and `shuffle_buffer=1` so the
          validation set is deterministic across runs. Metric comparisons
          between epochs are apples-to-apples.
        - `_BOTO3_CONFIG` is process-global. If you build two independent
          loaders against different buckets in the same process, they'll all
          share the last-set credentials — that's fine for the normal case
          (one bucket per run).
    """
    # Wire up credentials once for this process (workers inherit via fork).
    _BOTO3_CONFIG["access_key"] = (
        aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
    )
    _BOTO3_CONFIG["secret_key"] = (
        aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    _BOTO3_CONFIG["endpoint"] = endpoint
    _BOTO3_CONFIG["region"] = region
    if not _BOTO3_CONFIG["access_key"] or not _BOTO3_CONFIG["secret_key"]:
        raise RuntimeError(
            "No S3 credentials found. Set AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY, or pass them explicitly."
        )

    # List all shards (one-shot; ~13k keys, a few seconds)
    all_keys = _list_shards(s3_bucket)
    if not all_keys:
        raise RuntimeError(f"No .tar shards found in s3://{s3_bucket}/")
    print(f"[webdataset] Found {len(all_keys)} shards in s3://{s3_bucket}/")

    # Split at shard granularity (deterministic — shards were pre-shuffled).
    split = int(len(all_keys) * train_fraction)
    train_keys = all_keys[:split]
    val_keys = all_keys[split:]
    print(f"[webdataset]   train shards: {len(train_keys)}   "
          f"val shards: {len(val_keys)}")

    train_urls = [f"s3://{s3_bucket}/{k}" for k in train_keys]
    val_urls = [f"s3://{s3_bucket}/{k}" for k in val_keys]

    train_pipe = make_webdataset(
        train_urls,
        image_size=image_size,
        shuffle_shards=True,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
    )
    val_pipe = make_webdataset(
        val_urls,
        image_size=image_size,
        shuffle_shards=False,          # deterministic val order
        shuffle_buffer=1,              # no sample shuffle for val
        seed=seed,
    )

    # WebDataset pipelines yield ONE SAMPLE at a time. The DataLoader groups
    # those samples into batches, so `len(loader) == len(pipeline) // batch_size`
    # is applied automatically. We therefore pass `with_length(<sample count>)`
    # here — NOT batch count. Previous version passed batches, which caused
    # `len(train_loader)` to be `sample_count // batch_size // batch_size`,
    # i.e. 256× too small on a batch_size=256 run.
    train_len_samples = max(1, len(train_keys) * images_per_shard)
    val_len_samples   = max(1, len(val_keys)   * images_per_shard)

    if val_num_workers is None:
        val_num_workers = max(2, num_workers // 4)

    train_loader = DataLoader(
        train_pipe.with_length(train_len_samples),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_pipe.with_length(val_len_samples),
        batch_size=batch_size,
        num_workers=val_num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and val_num_workers > 0,
        prefetch_factor=prefetch_factor if val_num_workers > 0 else None,
    )

    return train_loader, val_loader


# ============================================================================
# Smoke test
# ============================================================================
if __name__ == "__main__":
    import time

    train_loader, val_loader = build_webdataset_loaders(
        s3_bucket="image-net-21k-shards",
        region="us-west-2",
        endpoint="https://s3.us-west-2.wasabisys.com",
        batch_size=32,
        image_size=256,
        num_workers=4,
    )

    print("Fetching a few batches to verify pipeline...")
    t0 = time.time()
    for i, (L, ab) in enumerate(train_loader):
        print(
            f"  batch {i}: L={tuple(L.shape)} ab={tuple(ab.shape)}  "
            f"L range=[{L.min():.3f},{L.max():.3f}]  "
            f"ab range=[{ab.min():.3f},{ab.max():.3f}]"
        )
        if i >= 4:
            break
    dt = time.time() - t0
    print(f"5 batches × 32 = 160 images in {dt:.1f}s  ({160/dt:.0f} img/s)")
