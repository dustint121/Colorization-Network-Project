# Image Colorization Project

Training scripts/loops for colorization of grayscale images using a U-Net built on a
pretrained ResNet-18 encoder, trained on several large image datasets
(landscapes, faces, and general-purpose ImageNet variants) and served from
S3-hosted data using both a per-object loader and a high-throughput
WebDataset shard loader.

## Purpose

Given a black-and-white (or desaturated) image, predict plausible color and
recombine it with the original luminance to produce a colorized image. This
project explores how dataset choice, model architecture, and data-loading
strategy affect colorization quality and training throughput — including
several dead ends and fixes that are documented in the FAQ below.

## Overview of the code

| File | Role |
|---|---|
| `class_ColorizationNet.py` | Original, simple encoder-decoder model (ResNet-18 encoder, plain upsampling decoder, no skip connections). Superseded by the U-Net below but kept for reference/comparison. |
| `unet_colorization.py` | Current model: `ColorizationUNet`, a ResNet-18 encoder with U-Net-style skip connections. This is what all recent training runs use. |
| `Image_Colorization_Project-fixed.ipynb` | Main notebook for **per-object S3 training** (`S3ColorizationDataset`, `build_train_val_s3_loaders`, `train_s3`, `VGGPerceptualLoss`). Use this for datasets stored as loose image files in S3 (with or without `train/`/`val/` subfolders). |
| `Image_Colorization_Project-webdataset.ipynb` (built by `build_wds_notebook.py`) | Notebook for **WebDataset shard training** (`train_wds`, `build_webdataset_loaders`). Use this for the ImageNet-21K shard pipeline — much higher throughput than per-object S3 reads. |
| `webdataset_colorization.py` | Standalone module version of the WebDataset loader (`build_webdataset_loaders`) — same logic as what's inlined into the WebDataset notebook. |
| `reshard_imagenet.py` | Offline tool that repackages a folder of loose ImageNet-21K JPEGs into shuffled `.tar` shards (WebDataset format) and uploads them to S3. Run this once per dataset before using the WebDataset notebook. |
| `run_model.py` | Inference script — loads a trained checkpoint and colorizes a single image on disk. |
| `apply_fixes.py` | Utility script used during development to patch notebook cells programmatically. |
| `build_wds_notebook.py` | Generator script that assembles `Image_Colorization_Project-webdataset.ipynb` cell-by-cell. Edit this file (not the notebook directly) when you need to change the WebDataset notebook, then re-run it to rebuild the `.ipynb`. |

### How to run it

**Per-object S3 dataset (loose files, optionally in `train/`/`val/` folders):**
```python
train_s3(
    s3_prefix="s3://your-bucket",
    region="us-west-2",
    endpoint="https://s3.us-west-2.wasabisys.com",
    num_epochs=8,
    batch_size=32,
    image_size=256,
    lr=1e-4,
    device="cuda",
)
```
`train_s3` calls `build_train_val_s3_loaders`, which lists all keys once,
auto-detects a `train`/`val` split from the key paths (see FAQ), and builds
both `DataLoader`s.

**WebDataset shards (recommended for large datasets like ImageNet-21K):**
```python
train_wds(
    s3_bucket="image-net-21k-shards",
    region="us-west-2",
    endpoint="https://s3.us-west-2.wasabisys.com",
    run_name="my_run",
    num_epochs=20,
    batch_size=256,
    image_size=256,
    num_workers=48,
    lr=1e-4,
    perceptual_weight=0.3,
    freeze_encoder_epochs=0,
    shuffle_buffer=8000,
    device=device,
    checkpoint_dir="checkpoints_wds",
    model_cls=ColorizationUNet,
    perceptual_cls=VGGPerceptualLoss,
    lab_to_rgb_fn=lab_to_rgb_torch,
)
```
Before this will work, the source images must first be resharded into `.tar`
files with `reshard_imagenet.py` and uploaded to the target bucket.

**Inference on a single image:**
```bash
python run_model.py --checkpoint checkpoints_wds/my_run_best.pt --image path/to/photo.jpg
```

Both notebooks save a `_best.pt` checkpoint (lowest validation loss so far)
and support resuming from it via a `resume_from=` argument.

## Datasets used

| Dataset | Size | Purpose / outcome |
|---|---|---|
| Same-image (sanity check) | 1,000 copies of one photo (Barack Obama) | Verification-only dataset to confirm the training loop can memorize a single image. It does — but the resulting model applies that one color scheme to every image, confirming it isn't learning general colorization (as expected). |
| Landscape (LHQ) — [Kaggle: dimensi0n/lhq-1024](https://www.kaggle.com/datasets/dimensi0n/lhq-1024?resource=download) | ~90,000 images | Landscape-only training. Colorizes landscape imagery reliably with no issues found in testing. Does not generalize to faces (expected — out of domain). |
| IMDb-Face — [GitHub: fwang91/IMDb-Face](https://github.com/fwang91/IMDb-Face?tab=readme-ov-file#data-download) | ~300,000 images of celebrities in varied settings | Face-focused training. Roughly a 50/50 hit rate on correct facial coloring. Successfully colorizes `test.jpg` (a training image), `old-lady.jpg`, and `group.jpg`, but fails on `black-man.jpg` — indicating a representation gap in the training set for some skin tones/lighting conditions rather than a bug. |
| ILSVRC (ImageNet-1K) — [image-net.org](https://www.image-net.org/download.php) | ~500,000 images across general categories | General-purpose "all-arounder" dataset. Coloring is often only "half-way" applied (partial/washed-out results) for reasons not fully diagnosed — see FAQ. Best validation L1 reached ≈0.0665 before overfitting set in (see FAQ). |
| ImageNet-21K — [image-net.org](https://www.image-net.org/download.php) | ~13.1 million images | Larger, more diverse general-purpose dataset; a more complete version of ILSVRC. Produces noticeably more complete/full coloring than ILSVRC. Requires the WebDataset reshard pipeline due to its size. Best validation L1 reached ≈0.0630, the lowest of any run so far. |

General finding: datasets trained on a narrow domain (landscapes) perform
better within that domain than the broad "all-arounder" datasets (ILSVRC,
ImageNet-21K) — the general-purpose models trade domain-specific accuracy for
broader (if inconsistent) coverage.

## Model architecture

**`ColorizationUNet`** (current model, in `unet_colorization.py`) — a U-Net
with a pretrained ResNet-18 encoder and skip connections, ~14M parameters.

```
INPUT: L (lightness channel, B×1×256×256, normalized to [0,1])
       → repeated to 3 channels to match ResNet's expected input
       ↓
ENCODER (pretrained ResNet-18, ImageNet weights)
  stem (conv+bn+relu)   → (B,  64, 128, 128)   ─┐ skip_1
  maxpool                → (B,  64,  64,  64)   │
  layer1                  → (B,  64,  64,  64)   ─┤ skip_2
  layer2                  → (B, 128,  32,  32)   ─┤ skip_3
  layer3                  → (B, 256,  16,  16)   ─┤ skip_4
  layer4 (bottleneck)     → (B, 512,   8,   8)
       ↓
DECODER (mirrors encoder, fuses skip connections via concatenation)
  up4: upsample + concat(skip_4) + DoubleConv → (B, 256, 16, 16)
  up3: upsample + concat(skip_3) + DoubleConv → (B, 128, 32, 32)
  up2: upsample + concat(skip_2) + DoubleConv → (B,  64, 64, 64)
  up1: upsample + concat(skip_1) + DoubleConv → (B,  32,128,128)
       ↓
  final 2x upsample → (B, 32, 256, 256)
  1x1 conv head     → (B,  2, 256, 256)
  tanh              → squashes to [-1, 1] to match ab/128 target range
       ↓
OUTPUT: ab (predicted A,B color channels, B×2×256×256)
```

**Major components:**

- **`DoubleConv`** — two conv+BN+ReLU layers; the standard U-Net building
  block used in every decoder stage.
- **`UpBlock`** — bilinear upsample (chosen over transposed convolution to
  avoid checkerboard artifacts) + skip-connection concatenation + `DoubleConv`.
  Skip connections let the decoder combine high-resolution spatial detail
  from the encoder (where things *are*) with deep semantic features from the
  bottleneck (what things *are*).
- **`set_encoder_trainable(bool)`** — freezes/unfreezes both `requires_grad`
  and BatchNorm behavior for all encoder layers at once. Used to freeze the
  pretrained ResNet-18 encoder for the first few epochs so the randomly
  initialized decoder can "catch up" before backpropagating into (and
  potentially corrupting) the pretrained features.
- **`VGGPerceptualLoss`** — a frozen, pretrained VGG-16 used to compare
  multi-layer features (edges → textures → parts → objects) between the
  predicted and ground-truth RGB reconstruction, added to the primary L1 loss
  with a configurable `perceptual_weight`. Encourages perceptually plausible
  color rather than just numerically close values.
- **`class_ColorizationNet.py` (legacy)** — the original, simpler model:
  same ResNet-18 encoder but a plain stacked-upsample decoder with no skip
  connections, ~11M parameters. Kept for comparison; produces blurrier output
  than the U-Net because it has no access to high-resolution encoder
  features during decoding.
- **Data representation** — all training operates in LAB color space rather
  than RGB: the network is given the L (lightness) channel and predicts the
  a/b (color) channels, which is the standard formulation for colorization
  because it cleanly separates luminance (already known) from the color
  information that actually needs to be predicted.

## FAQ — issues encountered and how they were addressed

**Q: Training silently ran for many epochs with `val_loss = inf`. What happened?**
The dataset had `train/` and `val/` subfolders nested inside the S3 bucket
(e.g. `train/ILSVRC2013_train/n00007846/...`), but the loader was probing for
files directly at `bucket/train/` and `bucket/val/`. That prefix matched
nothing, so both loaders silently built with 0 batches, and every epoch
"trained" and "validated" on zero samples, reporting `val_loss = inf`
sixteen times before being interrupted manually. Fixed by re-scanning the
*already-fetched* full key listing and classifying each key as train/val
based on whether any path segment is literally named `train`/`training` or
`val`/`valid`/`validation`, regardless of nesting depth. The loaders now
raise immediately if either split comes up empty, instead of training
silently on nothing.

**Q: `save_every_n_steps` didn't seem to do anything when set to 250 or 500 — training kept saving at the same old interval.**
The parameter only controlled a separate rolling-snapshot path; the actual
best-checkpoint save was still gated by a different, hardcoded interval. Fixed
by having `save_every_n_steps` directly control both the validation cadence
and the best-checkpoint save decision, and removing the old rolling-snapshot
code path entirely.

**Q: Resuming from a checkpoint printed `UserWarning: Detected call of lr_scheduler.step() before optimizer.step()`. Is this a real problem?**
No — it's a side effect of how the LR schedule was fast-forwarded on resume
(replaying `scheduler.step()` in a loop to catch up to the checkpoint's step
count, which calls it before the optimizer has stepped even once in the new
process). It didn't affect correctness, but the warning was noisy. Fixed by
directly setting the scheduler's internal step counters and current LR from
the checkpoint's `global_step`, which reaches the same mathematical state
without the warning.

**Q: Why did a later run appear to improve *more slowly* than an earlier run with the same settings — is this randomness?**
Not randomness — a scheduling artifact. The earlier run used a much smaller
`T_max` in its cosine LR schedule, so by the step where the "faster"
improvement was observed, the learning rate had already annealed to nearly
zero and the weights had effectively stopped moving, which looks like fast,
stable convergence but is actually the *model being frozen at a lucky point*.
The newer run had a `T_max` roughly 250× larger and was still at a much
higher LR at the same step, so its validation loss legitimately bounced
around while still learning. Both runs converged to the same ~0.068
validation L1 once given enough steps at low LR — confirming this was a
schedule-length artifact, not a regression.

**Q: Increasing `val_max_batches` (200 → 500) didn't change the validation curve. Why not?**
This ruled out "validation set too small / biased" as an explanation for
validation noise — it wasn't. The remaining noise came from genuinely bad
data in specific shards occasionally landing in the sampled validation
batches (see the outlier question below), not from an undersized eval set.

**Q: Does freezing the encoder for the first few epochs actually help?**
Yes, but the picture is nuanced. Freezing (`freeze_encoder_epochs > 0`) lets
the randomly initialized decoder adjust before touching the pretrained
ResNet-18 weights, and unfreezing at the right time (e.g. epoch 3–4) produced
a visible validation improvement in the same epoch it happened. However, on
smaller datasets (ILSVRC's ~450K images), keeping the encoder trainable for
too many additional epochs let the model overfit — validation loss started
rising again a few epochs after unfreezing while training loss kept falling.
Recommendation: unfreeze once, then watch validation closely for the
turnaround point rather than running a fixed large epoch count.

**Q: Occasional validation spikes (e.g. 0.09+ against a ~0.063 baseline) show up periodically in the logs. Bug?**
Not a bug — traced to specific "bad" shards in the WebDataset validation
sampling pool (a handful of shards that decode poorly or contain
lower-quality images) periodically landing in the fixed subset used for a
given validation pass. They don't affect which checkpoint gets saved as best
(a spike is never the lowest value) but they do inflate the visible variance
in the log. Left unaddressed since it doesn't affect training outcomes, only
log readability.

**Q: Is the model close to a plateau on validation loss?**
Depends on the run. On the ImageNet-21K WebDataset pipeline, dropping the
learning rate 10× (1e-3 → 1e-4) after an apparent plateau produced a further
~4% drop in validation L1 (0.0656 → 0.0630) over the following ~37,000 steps
— evidence the model still had headroom, and that the "plateau" was actually
an LR-noise floor, not a true capacity limit. On ILSVRC (a smaller dataset),
the model reached its best validation loss by epoch 6 and then overfit for
the remaining ~9 epochs (train loss kept falling, validation loss rose) —
there, more training was not the answer; the fix was to stop earlier or
freeze the encoder longer, not to adjust the learning rate.

**Q: Should `num_workers` be set higher than the machine's CPU count?**
PyTorch's own `DataLoader` warns when `num_workers` exceeds the system's
"suggested max" (roughly the physical CPU count), but for the S3-backed
datasets here this warning can be safely ignored and higher values are often
*better* despite it — these workers spend almost all their time blocked on
network I/O (fetching from S3/Wasabi) rather than CPU-bound decode work, so
running more workers than cores improves throughput by overlapping more
concurrent network requests, up to the point where the S3 endpoint itself
becomes the bottleneck. In practice, the WebDataset pipeline used
`num_workers=48` on a system reporting a suggested max of 12, without
slowness or freezing.

**Q: Why switch from per-object S3 reads to WebDataset shards at all?**
The per-object dataset (`S3ColorizationDataset`) issues one S3 GET request
per image, which is rate-limited by the S3-compatible endpoint (observed
around 640 GETs/sec on Wasabi) and further capped inside the DataLoader at
roughly 250 images/sec — a hard ceiling regardless of worker count or
instance size. `reshard_imagenet.py` repackages the dataset into ~1,000
images per `.tar` shard; each GET then retrieves an entire shard instead of
one image, cutting the effective request rate by roughly 1,000× and shifting
the bottleneck to CPU/decode throughput, which scales with `num_workers` as
expected. This is why ImageNet-21K training (13.1M images) uses the
WebDataset notebook exclusively — the per-object approach would not be
practical at that scale.

**Q: Why does ILSVRC output look "half-way" colorized while ImageNet-21K looks more complete?**
Not fully diagnosed. Both datasets were trained with the same architecture
and similar hyperparameters, and both converge to a similar validation L1
range (~0.063–0.067), so the difference isn't obviously a training bug.
The leading hypothesis is that ImageNet-21K's much larger class and scene
diversity (13.1M images vs. ILSVRC's ~500K) gives the encoder broader visual
priors to draw on, so it commits more confidently to color across a wider
range of unfamiliar inputs, whereas ILSVRC's narrower training distribution
leaves the model more conservative (partial/washed-out) on images outside
its narrower coverage. This has not been confirmed with a controlled
experiment.

**Q: Why does the model do so poorly on some faces (e.g. `black-man.jpg`) despite decent overall face-dataset performance?**
Likely a training-data representation gap in IMDb-Face for that particular
combination of skin tone, lighting, and pose, rather than an architecture or
training bug — the same model succeeds on other faces from both inside and
outside the training set (`test.jpg`, `old-lady.jpg`, `group.jpg`). Not yet
addressed; would require either augmenting the training set with more
diverse examples of the failing category or evaluating on a larger, more
systematic held-out face set to confirm the pattern before deciding on a fix.

## Key S3 / infrastructure notes

- Wasabi S3-compatible storage was used for the ImageNet-21K shards (region
  `us-west-2`), separate from whichever bucket hosts the per-object datasets.
- Mixed precision (`torch.amp.autocast` + `GradScaler`) is used throughout
  training for speed; loss values are always cast back to fp32 before
  logging/backprop to keep reported metrics accurate.
- Gradient clipping (max norm 10.0) is applied on every training step to
  guard against instability from the mixed-precision + perceptual-loss
  combination.
