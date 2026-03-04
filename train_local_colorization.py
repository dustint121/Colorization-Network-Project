# train_local_colorization.py
import os
import re
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from class_ColorizationNet import ColorizationNet
from build_loader_local import build_train_val_loaders


def find_latest_checkpoint(checkpoint_dir):
    """
    Look for files matching '*_colorization_step_<number>.pt' and
    return (path, step, epoch) for the highest step, or (None, 0, 0) if none.
    """
    if not os.path.isdir(checkpoint_dir):
        return None, 0, 0

    pattern = re.compile(r".*_colorization_step_(\d+)\.pt$")
    best_path = None
    best_step = 0
    best_epoch = 0

    for fname in os.listdir(checkpoint_dir):
        m = pattern.match(fname)
        if m:
            step = int(m.group(1))
            path = os.path.join(checkpoint_dir, fname)
            try:
                ckpt = torch.load(path, map_location="cpu")
                epoch = ckpt.get("epoch", 0)
            except Exception:
                continue
            if step > best_step:
                best_step = step
                best_path = path
                best_epoch = epoch

    return best_path, best_step, best_epoch


def train_local(
    image_root,
    num_epochs=5,
    batch_size=32,
    image_size=256,
    lr=1e-4,
    device="cuda",
    checkpoint_dir="checkpoints_local",
    save_every_steps=10_000,
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_loader, val_loader = build_train_val_loaders(
        root_dir=image_root,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=4,
        val_fraction=0.1,
    )

    model = ColorizationNet().to(device)
    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", patience=3, factor=0.5, verbose=True
                    )  # optional: reduce LR if val loss plateaus

    # Try to resume from latest checkpoint
    latest_ckpt, global_step, start_epoch = find_latest_checkpoint(checkpoint_dir)
    if latest_ckpt is not None:
        print(f"Resuming from checkpoint: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
    else:
        print("No checkpoint found, starting from scratch.")
        global_step = 0
        start_epoch = 0

    # Track best validation loss for "best" checkpoint saving
    best_val_loss = float("inf")

    for epoch in range(start_epoch, num_epochs):
        # ----- TRAIN -----
        model.train()
        epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch")

        for L_batch, ab_batch in epoch_bar:
            L_batch = L_batch.to(device).float()
            ab_batch = ab_batch.to(device).float()

            optimizer.zero_grad()
            ab_pred = model(L_batch)
            loss = criterion(ab_pred, ab_batch)
            loss.backward()

            # GRADIENT CLIPPING: clip gradients to max norm of 1.0 to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()

            global_step += 1
            loss_value = loss.item()
            epoch_bar.set_postfix({"loss": f"{loss_value:.4f}", "step": global_step})

            if global_step % save_every_steps == 0:
                # Regular step-based checkpoint
                step_name = f"Landscape Dataset 90K_colorization_step_{global_step}.pt"
                ckpt_path = os.path.join(checkpoint_dir, step_name)
                state = {
                    "step": global_step,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss": loss_value,
                }
                torch.save(state, ckpt_path)
                print(f"\nSaved checkpoint to {ckpt_path}")

        # ----- VALIDATION -----
        model.eval() # inference mode (no dropout/batchnorm updates)
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for L_val, ab_val in val_loader:
                L_val = L_val.to(device).float()
                ab_val = ab_val.to(device).float()
                ab_pred_val = model(L_val)
                loss_val = criterion(ab_pred_val, ab_val)
                val_loss_sum += loss_val.item() * L_val.size(0)
                val_count += L_val.size(0)

        mean_val_loss = val_loss_sum / max(1, val_count)
        print(f"Epoch {epoch+1}: val_loss = {mean_val_loss:.4f}")

        # LR SCHEDULER: step the ReduceLROnPlateau scheduler with the mean validation loss
        scheduler.step(mean_val_loss)

        # after computing mean_val_loss:
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_path = os.path.join(checkpoint_dir, "Landscape Dataset 90K_best.pt")
            torch.save(
                {
                    "step": global_step,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": mean_val_loss,
                },
                best_path,
            )
            print(f"New best checkpoint (val_loss={best_val_loss:.4f}) saved to {best_path}")


if __name__ == "__main__":
    image_root = "c:/Users/dusti/Downloads/landscape-images"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_local(
        image_root=image_root,
        num_epochs=5,
        batch_size=32,
        image_size=256,
        lr=1e-4,
        device=device,
        checkpoint_dir="checkpoints_local",
        save_every_steps=1_000,
    )


# check size of batches 
    # landscape dataset
        #train_loader = (90000 x 0.9) / 32 = 2531 batches per epoch
                        # 90000 / 32 = 2812 batches per epoch if no val split 
