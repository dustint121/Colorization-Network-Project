"""
Streaming WebDataset training loop for the colorization U-Net.

Used for ImageNet-21K colorization training; 
13.1 million images is too much too read into memory individually due to Time to First Byte (TTFB) and overall memory constraints; 
hence, streaming via WebDataset is employed where "shards" of 1000 images are read sequentially in the form of tar files.
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from webdataset_colorization import build_webdataset_loaders


# ============================================================================
# Module-level knobs
# ============================================================================

_LR_MIN                = 1e-6      # cosine schedule floor
_GRAD_CLIP             = 10.0      # torch.nn.utils.clip_grad_norm_ bound
_LOG_EVERY_N_STEPS     = 50        # tqdm postfix refresh cadence
_TRAIN_SHARD_FRACTION  = 0.95      # rest = val
_IMAGES_PER_SHARD      = 1000      # only affects estimated epoch length

# Val cadence fallback default when the user passes neither
# `save_every_n_steps` nor an obvious override. Not exposed as a param.
_VAL_DEFAULT_CAP_STEPS = 500


def train_wds(
    # ---- data ----
    s3_bucket: str,
    region: str,
    endpoint: str,
    shuffle_buffer: int = 4000,
    # ---- optimization ----
    num_epochs: int = 30,
    batch_size: int = 64,
    image_size: int = 256,
    lr: float = 5e-4,
    freeze_encoder_epochs: int = 3,
    perceptual_weight: float = 0.3,
    # ---- infra ----
    num_workers: int = 8,
    device: str = "cuda",
    checkpoint_dir: str = "checkpoints_wds",
    run_name: str = "imagenet21k",
    # ---- resume / val cadence ----
    resume_from: Optional[str] = None,
    save_every_n_steps: Optional[int] = None,
    val_max_batches: Optional[int] = 200,
    # ---- injected classes (kept out of import graph on purpose) ----
    model_cls=None,               # e.g. ColorizationUNet
    perceptual_cls=None,          # e.g. VGGPerceptualLoss
    lab_to_rgb_fn=None,           # e.g. lab_to_rgb_torch from the notebook
):
    """Train the colorization U-Net over a WebDataset stream.

    Key knobs:
        save_every_n_steps: how often to run validation AND save-if-best.
            Not an unconditional per-step snapshot. Default = min(500,
            steps_per_epoch // 20). Set explicitly to override.
        val_max_batches: cap on val loop length (batches). Default 200 keeps
            each val fast (~30s). Pass None to use the full val set.
        resume_from: path to a checkpoint. Restores model + optimizer +
            AMP scaler + best_val + step, and re-aligns the cosine schedule
            to the resumed step without stepping the optimizer.

    Returns:
        dict with final metrics: {"best_val_loss", "best_ckpt",
        "final_ckpt", "final_step", "final_epoch"}
    """
    if model_cls is None or perceptual_cls is None or lab_to_rgb_fn is None:
        raise ValueError(
            "Pass model_cls=ColorizationUNet, perceptual_cls=VGGPerceptualLoss, "
            "and lab_to_rgb_fn=lab_to_rgb_torch."
        )

    # ------------------------------------------------------------------------
    # 0) Warning suppression
    # ------------------------------------------------------------------------
    # The training loop produces a small handful of harmless-but-loud warnings
    # every val pass

    # We silence them at the source so training logs stay signal-heavy.
    warnings.filterwarnings(
        "ignore",
        message="Corrupt EXIF data.*",
    )
    warnings.filterwarnings(
        "ignore",
        message="Truncated File Read",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*can only test a child process.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*with_length\\(\\) only sets the value of __len__.*",
    )
    # Skip-bad-sample warnings from our own decoder are noisy but low-value
    # once you know the dataset has ~a few thousand corrupt JPEGs. Suppress.
    warnings.filterwarnings(
        "ignore",
        message=r"\[webdataset\] skipping bad sample.*",
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    device_t = torch.device(device)

    # ------------------------------------------------------------------------
    # 1) Data
    # ------------------------------------------------------------------------
    print(f"[train_wds] building WebDataset loaders "
          f"(bucket={s3_bucket}, image_size={image_size}, "
          f"batch_size={batch_size}, num_workers={num_workers})")
    train_loader, val_loader = build_webdataset_loaders(
        s3_bucket=s3_bucket,
        region=region,
        endpoint=endpoint,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        train_fraction=_TRAIN_SHARD_FRACTION,
        shuffle_buffer=shuffle_buffer,
        images_per_shard=_IMAGES_PER_SHARD,
    )
    steps_per_epoch = len(train_loader)          # from with_length(...)
    total_steps = steps_per_epoch * num_epochs
    print(f"[train_wds]   steps/epoch = {steps_per_epoch:,}   "
          f"total steps = {total_steps:,}")

    # Val cadence: run validation and save-if-best every `check_every_n_steps`
    # training steps. On big datasets (47k steps/epoch), the default gives
    # ~20 checks per epoch capped at every 500 steps.
    if save_every_n_steps is not None:
        check_every_n_steps = save_every_n_steps
    else:
        check_every_n_steps = min(
            _VAL_DEFAULT_CAP_STEPS,
            max(1, steps_per_epoch // 20),
        )
    print(f"[train_wds]   checking val + saving best every "
          f"{check_every_n_steps:,} steps")

    # Val loop cap. Full val on ImageNet-21K is ~2500 batches (~5 min at bs=256);
    # 200 batches ≈ 50K images and takes ~30 s, plenty for a stable val_L1
    # estimate. Pass val_max_batches=None to use the full val set.
    val_batches_total = len(val_loader) if val_max_batches is None \
                        else min(val_max_batches, len(val_loader))

    # ------------------------------------------------------------------------
    # 2) Model, loss, optimizer, scaler, scheduler
    # ------------------------------------------------------------------------
    model = model_cls(pretrained=True,
                      freeze_encoder_epochs=freeze_encoder_epochs).to(device_t)
    criterion = nn.L1Loss()
    perceptual = perceptual_cls(device=device).to(device_t)

    optimizer = Adam(model.parameters(), lr=lr)
    scaler = GradScaler("cuda", enabled=(device_t.type == "cuda"))
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=_LR_MIN)

    # ------------------------------------------------------------------------
    # 3) Optionally resume from a checkpoint
    # ------------------------------------------------------------------------
    # We restore model + optimizer + AMP scaler + counters, then align the
    # cosine LR schedule to the resumed step by SETTING `last_epoch` and
    # recomputing the LR directly. The old "for _ in range(N): scheduler.step()"
    # trick works but (a) is O(N) at the start of every resume, and (b) causes
    # PyTorch to warn about scheduler.step() being called before optimizer.step()
    # on the very first resumed iteration. Setting `last_epoch` sidesteps both.
    start_epoch = 0
    global_step = 0
    best_val = float("inf")
    if resume_from is not None and os.path.exists(resume_from):
        print(f"[train_wds] resuming from {resume_from}")
        ckpt = torch.load(resume_from, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        if "optim_state" in ckpt:
            optimizer.load_state_dict(ckpt["optim_state"])
        if "scaler_state" in ckpt and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        best_val = ckpt.get("best_val_loss", float("inf"))

        # Align cosine schedule to the resumed step in O(1).
        # CosineAnnealingLR internally reads `last_epoch` and computes the LR
        # from it; setting it directly (with `_step_count` bumped past 1 so
        # PyTorch's "step before optimizer" guard stays quiet) is equivalent
        # to N calls to scheduler.step() but instant and warning-free.
        scheduler.last_epoch = global_step - 1
        scheduler._step_count = global_step + 1
        for pg, lr_new in zip(optimizer.param_groups,
                              scheduler.get_last_lr()):
            pg["lr"] = lr_new

        print(f"[train_wds]   resumed at epoch={start_epoch}  step={global_step}  "
              f"best_val={best_val:.5f}  lr={scheduler.get_last_lr()[0]:.2e}")

    # ------------------------------------------------------------------------
    # 4) Encoder freeze/unfreeze
    # ------------------------------------------------------------------------
    # ColorizationUNet.set_encoder_trainable(bool) handles both requires_grad
    # AND BatchNorm eval() toggling on the encoder (so running stats don't
    # drift during the freeze phase). We just delegate.
    model.set_encoder_trainable(freeze_encoder_epochs == 0)

    # ------------------------------------------------------------------------
    # 5) Val loop (used at intervals during training)
    # ------------------------------------------------------------------------
    def run_validation() -> float:
        model.eval()
        losses = []
        n_batches = 0
        with torch.no_grad():
            for L, ab in val_loader:
                L = L.to(device_t, non_blocking=True)
                ab = ab.to(device_t, non_blocking=True)
                ab_pred = model(L)
                losses.append(criterion(ab_pred, ab).item())
                n_batches += 1
                if n_batches >= val_batches_total:
                    break
        model.train()
        # model.train() flips every child back into train mode, including the
        # encoder BN layers we deliberately put in eval() during the freeze.
        # Re-apply the freeze if we're still in that phase.
        if global_step // max(1, steps_per_epoch) < freeze_encoder_epochs:
            model.set_encoder_trainable(False)
        return sum(losses) / max(1, len(losses))

    # ------------------------------------------------------------------------
    # 6) Training loop
    # ------------------------------------------------------------------------
    model.train()
    if freeze_encoder_epochs > 0:
        model.set_encoder_trainable(False)

    best_ckpt = os.path.join(checkpoint_dir, f"{run_name}_best.pt")

    for epoch in range(start_epoch, num_epochs):
        # Unfreeze at the epoch boundary.
        if epoch == freeze_encoder_epochs and freeze_encoder_epochs > 0:
            print(f"[train_wds] epoch {epoch}: UNFREEZING encoder")
            model.set_encoder_trainable(True)

        epoch_loss_sum = 0.0
        epoch_batches = 0
        t_epoch = time.time()
        pbar = tqdm(
            train_loader,
            total=steps_per_epoch,
            desc=f"epoch {epoch}",
            unit="batch",
        )

        # WebDataset is an IterableDataset. `with_length()` only sets what tqdm
        # displays; it does NOT cause the iterator to stop after N batches.
        # We break the loop manually once we've processed steps_per_epoch, else
        # "epoch 0" runs forever and no epoch-boundary logic (unfreeze, epoch
        # checkpoint) ever fires.
        steps_this_epoch = 0
        for L, ab in pbar:
            L = L.to(device_t, non_blocking=True)
            ab = ab.to(device_t, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Forward under autocast; loss compute in fp32 to avoid
            # perceptual-loss underflow on small ab errors.
            with autocast(device_type=device_t.type, enabled=(device_t.type == "cuda")):
                ab_pred = model(L)

            ab_pred_f32 = ab_pred.float()
            L_f32 = L.float()
            ab_f32 = ab.float()
            loss_l1 = criterion(ab_pred_f32, ab_f32)
            if perceptual_weight > 0:
                # VGGPerceptualLoss expects two 3-channel RGB tensors in [0,1].
                # Reconstruct RGB from (L, ab_pred) / (L, ab_true) via the
                # differentiable lab_to_rgb helper so grads flow through.
                pred_rgb = lab_to_rgb_fn(L_f32, ab_pred_f32)
                with torch.no_grad():
                    target_rgb = lab_to_rgb_fn(L_f32, ab_f32)
                loss_perc = perceptual(pred_rgb, target_rgb)
                loss = loss_l1 + perceptual_weight * loss_perc
            else:
                loss_perc = torch.tensor(0.0, device=device_t)
                loss = loss_l1

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), _GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss_sum += loss_l1.item()   # L1 only, comparable to old runs
            epoch_batches += 1
            global_step += 1

            if global_step % _LOG_EVERY_N_STEPS == 0:
                with torch.no_grad():
                    pred_abs = ab_pred_f32.abs().mean().item()
                    tgt_abs = ab.float().abs().mean().item()
                    chroma_ratio = pred_abs / max(tgt_abs, 1e-8)
                pbar.set_postfix(
                    {
                        "L1": f"{loss_l1.item():.5f}",
                        "perc": f"{loss_perc.item():.4f}" if perceptual_weight > 0 else "off",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                        "chroma": f"{chroma_ratio:.2f}",
                    }
                )

            # Val + save-if-best. This is the ONLY checkpoint write path in
            # the loop; `save_every_n_steps` controls this cadence.
            if global_step % check_every_n_steps == 0:
                val_loss = run_validation()
                improved = val_loss < best_val
                mark = "  ✓NEW BEST" if improved else ""
                print(f"[val] step={global_step} epoch={epoch} "
                      f"val_L1={val_loss:.5f}  best={best_val:.5f}{mark}")
                if improved:
                    best_val = val_loss
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "optim_state": optimizer.state_dict(),
                            "scaler_state": (scaler.state_dict()
                                             if scaler.is_enabled() else None),
                            "epoch": epoch,
                            "global_step": global_step,
                            "best_val_loss": best_val,
                            "image_size": image_size,
                        },
                        best_ckpt,
                    )
                    print(f"[val]   saved best to {best_ckpt}")

            steps_this_epoch += 1
            if steps_this_epoch >= steps_per_epoch:
                # Reached epoch boundary; break so the outer loop advances
                # and unfreeze / epoch bookkeeping can fire.
                break

        pbar.close()
        dt = time.time() - t_epoch
        avg_l1 = epoch_loss_sum / max(1, epoch_batches)
        print(f"[epoch {epoch}] avg_L1 = {avg_l1:.5f}   "
              f"batches = {epoch_batches}   wall = {dt/60:.1f} min")

    # Final val + checkpoint.
    final_val = run_validation()
    final_ckpt = os.path.join(checkpoint_dir, f"{run_name}_final.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
            "epoch": num_epochs - 1,
            "global_step": global_step,
            "best_val_loss": best_val,
            "image_size": image_size,
        },
        final_ckpt,
    )
    print(f"[train_wds] DONE.  final_val_L1={final_val:.5f}  "
          f"best_val_L1={best_val:.5f}")
    print(f"[train_wds]   best checkpoint:  {best_ckpt}")
    print(f"[train_wds]   final checkpoint: {final_ckpt}")

    return {
        "best_val_loss": best_val,
        "best_ckpt": best_ckpt,
        "final_ckpt": final_ckpt,
        "final_step": global_step,
        "final_epoch": num_epochs - 1,
    }
