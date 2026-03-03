# train_local_colorization.py
import os
import re
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from class_ColorizationNet import ColorizationNet
from build_loader_local import build_local_loader


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

    loader = build_local_loader(
        root_dir=image_root,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=4,
        shuffle=True,
    )

    model = ColorizationNet().to(device)
    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=lr)

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

    # Track best loss seen so far (for "best" checkpoint)
    best_loss = float("inf")

    for epoch in range(start_epoch, num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs} (global_step={global_step})...")
        epoch_bar = tqdm(loader, desc=f"Epoch {epoch+1}", unit="batch")

        for L_batch, ab_batch in epoch_bar:
            L_batch = L_batch.to(device).float()
            ab_batch = ab_batch.to(device).float()

            optimizer.zero_grad()
            ab_pred = model(L_batch)
            loss = criterion(ab_pred, ab_batch)
            loss.backward()
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

                # "Best" checkpoint (based on lowest loss so far)
                if loss_value < best_loss:
                    best_loss = loss_value
                    best_path = os.path.join(
                        checkpoint_dir, "Landscape Dataset 90K_best.pt"
                    )
                    torch.save(state, best_path)
                    print(
                        f"New best checkpoint (loss={best_loss:.4f}) saved to {best_path}"
                    )


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
